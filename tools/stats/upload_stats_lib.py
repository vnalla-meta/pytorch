import gzip
import io
import json
import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from warnings import warn

import boto3  # type: ignore[import]
import requests
import rockset  # type: ignore[import]

from torch.testing._internal.common_utils import IS_CI

PYTORCH_REPO = "https://api.github.com/repos/pytorch/pytorch"
S3_RESOURCE = boto3.resource("s3")
TARGET_WORKFLOW = "--rerun-disabled-tests"


def _get_request_headers() -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + os.environ["GITHUB_TOKEN"],
    }


def _get_artifact_urls(prefix: str, workflow_run_id: int) -> Dict[Path, str]:
    """Get all workflow artifacts with 'test-report' in the name."""
    response = requests.get(
        f"{PYTORCH_REPO}/actions/runs/{workflow_run_id}/artifacts?per_page=100",
    )
    artifacts = response.json()["artifacts"]
    while "next" in response.links.keys():
        response = requests.get(
            response.links["next"]["url"], headers=_get_request_headers()
        )
        artifacts.extend(response.json()["artifacts"])

    artifact_urls = {}
    for artifact in artifacts:
        if artifact["name"].startswith(prefix):
            artifact_urls[Path(artifact["name"])] = artifact["archive_download_url"]
    return artifact_urls


def _download_artifact(
    artifact_name: Path, artifact_url: str, workflow_run_attempt: int
) -> Path:
    # [Artifact run attempt]
    # All artifacts on a workflow share a single namespace. However, we can
    # re-run a workflow and produce a new set of artifacts. To avoid name
    # collisions, we add `-runattempt1<run #>-` somewhere in the artifact name.
    #
    # This code parses out the run attempt number from the artifact name. If it
    # doesn't match the one specified on the command line, skip it.
    atoms = str(artifact_name).split("-")
    for atom in atoms:
        if atom.startswith("runattempt"):
            found_run_attempt = int(atom[len("runattempt") :])
            if workflow_run_attempt != found_run_attempt:
                print(
                    f"Skipping {artifact_name} as it is an invalid run attempt. "
                    f"Expected {workflow_run_attempt}, found {found_run_attempt}."
                )

    print(f"Downloading {artifact_name}")

    response = requests.get(artifact_url, headers=_get_request_headers())
    with open(artifact_name, "wb") as f:
        f.write(response.content)
    return artifact_name


def download_s3_artifacts(
    prefix: str, workflow_run_id: int, workflow_run_attempt: int
) -> List[Path]:
    bucket = S3_RESOURCE.Bucket("gha-artifacts")
    objs = bucket.objects.filter(
        Prefix=f"pytorch/pytorch/{workflow_run_id}/{workflow_run_attempt}/artifact/{prefix}"
    )

    found_one = False
    paths = []
    for obj in objs:
        found_one = True
        p = Path(Path(obj.key).name)
        print(f"Downloading {p}")
        with open(p, "wb") as f:
            f.write(obj.get()["Body"].read())
        paths.append(p)

    if not found_one:
        print(
            "::warning title=s3 artifacts not found::"
            "Didn't find any test reports in s3, there might be a bug!"
        )
    return paths


def download_gha_artifacts(
    prefix: str, workflow_run_id: int, workflow_run_attempt: int
) -> List[Path]:
    artifact_urls = _get_artifact_urls(prefix, workflow_run_id)
    paths = []
    for name, url in artifact_urls.items():
        paths.append(_download_artifact(Path(name), url, workflow_run_attempt))
    return paths


def upload_to_rockset(
    collection: str, docs: List[Any], workspace: str = "commons"
) -> None:
    print(f"Writing {len(docs)} documents to Rockset")
    client = rockset.RocksetClient(
        host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )
    client.Documents.add_documents(
        collection=collection,
        data=docs,
        workspace=workspace,
    )
    print("Done!")


def emit_metric(
    metric_name: str,
    metrics: Dict[str, Any],
) -> None:
    """
    Upload a metric to DynamoDB (and from there, Rockset).

    Parameters:
        metric_name:
            Name of the metric. Every unique metric must have a different name
            and must be emitted just once per run attempt.
        metrics: The actual data to record.

    Some default values are populated from environment variables, which must be set
    for metrics to be emitted. (If they're not set, this function becomes a noop):

    Env vars which should be set:
        GITHUB_REPOSITORY: The repo name, e.g. "pytorch/pytorch"
        GITHUB_WORKFLOW: The workflow name
        GITHUB_JOB: The job id
        GITHUB_RUN_NUMBER: The workflow run number
        GITHUB_RUN_ATTEMPT: The workflow run attempt
    """

    if not IS_CI:
        return  # Don't emit metrics if we're not running in CI

    if metrics is None:
        raise ValueError("You didn't ask to upload any metrics!")

    # Env vars that we use to determine basic info about the workflow run
    input_via_env_vars = {
        "repo": "GITHUB_REPOSITORY",
        "workflow": "GITHUB_WORKFLOW",
        "job": "GITHUB_JOB",
        "workflow_run_number": "GITHUB_RUN_NUMBER",
        "workflow_run_attempt": "GITHUB_RUN_ATTEMPT",
    }

    reserved_metric_keys = [
        "dynamo_key",
        "metric_name",
        *input_via_env_vars.keys(),
    ]

    # Ensure the metrics dict doesn't contain these reserved keys
    for key in reserved_metric_keys:
        used_reserved_keys = [k for k in metrics.keys() if k == key]
        if used_reserved_keys in metrics:
            raise ValueError(f"Metrics dict contains reserved keys [{', '.join(key)}]")

    reserved_env_metrics = []
    # Ensure we have a value for all the required env var based metrics
    for var, env_var in input_via_env_vars.items():
        reserved_env_metrics[var] = os.environ[env_var]
        if not reserved_env_metrics[var]:
            warn(
                f"Not emitting metrics, missing {var}. Please set the {env_var} environment variable to pass in this value."
            )
            return

    dynamo_key = "/".join([
        reserved_env_metrics["repo"],
        metric_name,
        reserved_env_metrics["workflow"],
        reserved_env_metrics["job"],
        reserved_env_metrics["workflow_run_number"],
        reserved_env_metrics["workflow_run_attempt"],
    ])

    try:
        boto3.resource("dynamodb").Table("torchci-metrics").put_item(
            Item={
                "dynamo_key": dynamo_key,
                "metric_name": metric_name,
                **reserved_env_metrics,
                **metrics,
            }
        )
    except Exception as e:
        # We don't want to fail the job if we can't upload the metric.
        # We still raise the ValueErrors outside this try block since those indicate improperly configured metrics
        warn(f"Error uploading metric to DynamoDB: {e}")
        return


def upload_to_s3(
    bucket_name: str,
    key: str,
    docs: List[Dict[str, Any]],
) -> None:
    print(f"Writing {len(docs)} documents to S3")
    body = io.StringIO()
    for doc in docs:
        json.dump(doc, body)
        body.write("\n")

    S3_RESOURCE.Object(
        f"{bucket_name}",
        f"{key}",
    ).put(
        Body=gzip.compress(body.getvalue().encode()),
        ContentEncoding="gzip",
        ContentType="application/json",
    )
    print("Done!")


def read_from_s3(
    bucket_name: str,
    key: str,
) -> List[Dict[str, Any]]:
    print(f"Reading from s3://{bucket_name}/{key}")
    body = (
        S3_RESOURCE.Object(
            f"{bucket_name}",
            f"{key}",
        )
        .get()["Body"]
        .read()
    )
    results = gzip.decompress(body).decode().split("\n")
    return [json.loads(result) for result in results if result]


def upload_workflow_stats_to_s3(
    workflow_run_id: int,
    workflow_run_attempt: int,
    collection: str,
    docs: List[Dict[str, Any]],
) -> None:
    bucket_name = "ossci-raw-job-status"
    key = f"{collection}/{workflow_run_id}/{workflow_run_attempt}"
    upload_to_s3(bucket_name, key, docs)


def upload_file_to_s3(
    file_name: str,
    bucket: str,
    key: str,
) -> None:
    """
    Upload a local file to S3
    """
    print(f"Upload {file_name} to s3://{bucket}/{key}")
    boto3.client("s3").upload_file(
        file_name,
        bucket,
        key,
    )


def unzip(p: Path) -> None:
    """Unzip the provided zipfile to a similarly-named directory.

    Returns None if `p` is not a zipfile.

    Looks like: /tmp/test-reports.zip -> /tmp/unzipped-test-reports/
    """
    assert p.is_file()
    unzipped_dir = p.with_name("unzipped-" + p.stem)
    print(f"Extracting {p} to {unzipped_dir}")

    with zipfile.ZipFile(p, "r") as zip:
        zip.extractall(unzipped_dir)


def is_rerun_disabled_tests(root: ET.ElementTree) -> bool:
    """
    Check if the test report is coming from rerun_disabled_tests workflow
    """
    skipped = root.find(".//*skipped")
    # Need to check against None here, if not skipped doesn't work as expected
    if skipped is None:
        return False

    message = skipped.attrib.get("message", "")
    return TARGET_WORKFLOW in message or "num_red" in message
