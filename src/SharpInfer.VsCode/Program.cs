namespace SharpInfer.VsCode;

/// <summary>
/// Entry point for the VS Code extension bridge process.
/// Communicates with the VS Code extension via JSON-RPC over stdin/stdout.
///
/// The companion VS Code extension (TypeScript) spawns this process and
/// sends/receives messages. A minimal extension.ts would look like:
///
///   const child = spawn('dotnet', ['run', '--project', 'SharpInfer.VsCode']);
///   // Send: child.stdin.write(jsonRpcMessage)
///   // Receive: child.stdout.on('data', handler)
/// </summary>
public class Program
{
    public static async Task Main(string[] args)
    {
        using var bridge = new LanguageServerBridge(
            Console.OpenStandardInput(),
            Console.OpenStandardOutput());

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

        await bridge.RunAsync(cts.Token);
    }
}
