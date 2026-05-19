# Setting Up a Self-Hosted ADO Agent

This guide walks through provisioning a Windows machine as a self-hosted Azure DevOps (ADO) agent for the `modelkit-selfhost-pool`, configuring power settings so the agent stays online, and registering the agent to auto-start at logon.

References:

- Pool: <https://dev.azure.com/microsoft/windows.ai.toolkit/_settings/agentqueues?queueId=580593&view=agents>
- 1ES guide — Register a self-hosted agent without a PAT: <https://eng.ms/docs/coreai/devdiv/one-engineering-system-1es/1es-docs/1es-security-configuration/configuration-guides/register-self-hosted-agent-without-pat>

---

## Step 1: Download & Configure the Agent

1. Follow the "New agent" instructions on the [pool's agents page](https://dev.azure.com/microsoft/windows.ai.toolkit/_settings/agentqueues?queueId=580593&view=agents) to download the agent zip and extract it to `C:\agent`.

2. From the `C:\agent` directory, sign in to Azure and configure the agent. Replace `NPU-OV` with the desired agent name (e.g., `NPU-OV`, `GPU-DML`, etc.):

    ```powershell
    az login

    $env:VSTS_AGENT_INPUT_TOKEN = az account get-access-token `
      --resource 499b84ac-1321-427f-aa17-267ca6975798 `
      --query accessToken --output tsv

    .\config.cmd --unattended `
      --url https://dev.azure.com/microsoft `
      --pool modelkit-selfhost-pool `
      --agent NPU-OV `
      --auth pat `
      --acceptTeeEula

    Remove-Item Env:VSTS_AGENT_INPUT_TOKEN
    ```

    > The `--resource` GUID is the Azure DevOps resource ID and is the same for all tenants.

---

## Step 2: Prevent the Network Adapter from Sleeping

If the OS turns off the network adapter to save power, the agent will go offline. Disable this behavior:

1. Open **Device Manager**.
2. Expand **Network adapters**, right-click your active adapter, and choose **Properties**.
3. Go to the **Power Management** tab.
4. Uncheck **Allow the computer to turn off this device to save power**.
5. Click **OK**.

Repeat for each network adapter the machine uses to reach Azure DevOps.

---

## Step 3: Register the Agent to Auto-Start at Logon

Use the provided PowerShell script to register a Scheduled Task that launches `C:\agent\run.cmd` at user logon with a visible console window. The script self-elevates via UAC if needed.

**Register:**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\e2e_eval\setup_ado_agent.ps1
```

**Unregister:**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\e2e_eval\setup_ado_agent.ps1 -Unregister
```

After registration, sign out and sign back in to verify the agent console launches automatically and the agent appears as **Online** in the pool.
