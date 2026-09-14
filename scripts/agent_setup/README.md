# Setting Up a Self-Hosted ADO Agent

This guide walks through provisioning a Windows machine as a self-hosted Azure DevOps (ADO) agent for the `modelkit-selfhost-pool`, configuring power settings so the agent stays online, registering the agent to auto-start at logon, and (for DirectML GPU agents) keeping the GPU available after RDP disconnect.

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
powershell -ExecutionPolicy Bypass -File .\scripts\agent_setup\setup_ado_agent.ps1
```

**Unregister:**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\agent_setup\setup_ado_agent.ps1 -Unregister
```

After registration, sign out and sign back in to verify the agent console launches automatically and the agent appears as **Online** in the pool.

---

## Step 4: Keep the GPU Available After RDP Disconnect (DirectML agents only)

**Only needed on GPU agents that run DirectML (`GPU-DML`) and are accessed over Remote Desktop.** Skip this on CPU/NPU/OpenVINO-only agents.

RDP session transitions on a shared GPU agent can leave stale Remote Display Adapter nodes and duplicate GPU routes. DirectML may then select an unusable route even while the physical GPU remains available. Symptoms include DML device-creation errors and mismatches between ORT and DXCore adapter inventories.

The script registers a SYSTEM Scheduled Task (`KeepSessionOnConsole`) triggered by TerminalServices-LocalSessionManager **Event ID 24**. Its enhanced worker matches the version deployed on the shared development agent:

- A remote IP source triggers immediate handling; local/loopback transitions are ignored to avoid disrupting reconnects. Older tasks that only supply a SessionId use a matching Event 24 from the last minute to resolve the source address.
- Session 1 is eligible; only service Session 0 is excluded. A reconnected session or another user's occupied console is left alone.
- The worker attempts `tscon` up to three times, waiting two seconds between state checks, and verifies that the target reached console.
- Only after verification, it removes non-present RDP display nodes (`SWD\REMOTEDISPLAYENUM\*`, phantom/error 45). Physical PCI GPUs and present RDP devices are excluded.
- The optional `-CleanupOnly` mode runs that same cleanup only when a console session exists.

Console recovery and PnP cleanup **do not prove DML readiness** or remove every stale ORT/DXGI route. The worker logs warnings and retains its deployed exit-zero behavior even on errors. Use the separate physical-GPU pin in the E2E pipeline to avoid known duplicate routes, and verify with real inference.

**Register** (run once, elevated — the script reports an error if not elevated):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\agent_setup\setup_rdp_gpu_keepalive.ps1
```

The installer writes the worker to `C:/agent/tools/keep_console.ps1` and logs activity to `C:/agent/tools/keep_console.log`. Re-registering passes both the event SessionId and source address. Its smoke test omits SessionId, so it performs no redirect or device cleanup. To verify recovery during a planned RDP disconnect, check for a verified console transition and review cleanup warnings, then run a DML inference check. Merely editing this branch does not update the installed worker or task.

**Unregister:**

```powershell
Unregister-ScheduledTask -TaskName KeepSessionOnConsole -Confirm:$false
```
