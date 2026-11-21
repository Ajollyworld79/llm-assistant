from railway_sdk import define_railway, github, project, service, volume

# This repository manages only its own service in the shared project. Other
# repositories export their own partial name.
# See https://docs.railway.com/infrastructure-as-code#multi-repo-projects
PARTIAL = "llm-assistant"


@define_railway
def main(ctx=None):
    # Migrated from railway.json (Config as Code stops being read 2026-12-01):
    # Dockerfile build and restart retries carried over,
    # everything else mirrors the live service.
    data_volume = volume(
        "llm-assistant-volume",
        {
            "alerts": {"usage": {"80": {}, "95": {}, "100": {}}},
            "allowOnlineResize": True,
            "region": "europe-west4-drams3a",
            "sizeMB": 50000,
        },
    )
    app = service(
        "llm-assistant",
        source=github("Ajollyworld79/llm-assistant", checkSuites=False),
        build={"buildEnvironment": "V3", "builder": "DOCKERFILE", "dockerfilePath": "Dockerfile"},
        deploy={"restartPolicyMaxRetries": 10},
        replicas={"europe-west4-drams3a": 1},
        volumeMounts={"/app/qdrant_data": data_volume},
        groupId="LLm-demo",
    )
    # Variables stay owned by Railway (dashboard); this file never touches them.
    return project("responsible-insight", variables={"managed": False}, resources=[data_volume, app])
