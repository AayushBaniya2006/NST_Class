"""Upload a trained model to Vertex AI Model Registry."""
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Upload model to Vertex AI")
    parser.add_argument("--model-path", type=str, required=True, help="GCS path to model artifact directory")
    parser.add_argument("--display-name", type=str, required=True, help="Model display name")
    parser.add_argument("--project", type=str, required=True, help="GCP project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region")
    parser.add_argument("--serving-container", type=str,
                        default="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
                        help="Serving container image URI")
    args = parser.parse_args()

    from google.cloud import aiplatform

    aiplatform.init(project=args.project, location=args.region)

    model = aiplatform.Model.upload(
        display_name=args.display_name,
        artifact_uri=args.model_path,
        serving_container_image_uri=args.serving_container,
    )

    logger.info(f"Model uploaded: {model.resource_name}")
    logger.info(f"Model ID: {model.name}")
    print(f"\nVertex Model ID: {model.resource_name}")


if __name__ == "__main__":
    main()
