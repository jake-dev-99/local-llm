import * as vscode from 'vscode';
import { describeError, errorStack } from './errorDetail';

export class LocalLlmLogger implements vscode.Disposable {
  private readonly channel = vscode.window.createOutputChannel('Local LLM');

  constructor(private level: 'error' | 'info' | 'debug') {}

  setLevel(level: 'error' | 'info' | 'debug'): void {
    this.level = level;
  }

  /**
   * Failures are always reported in full, at every log level.
   *
   * The cause chain and the stack are the difference between a report that can be
   * acted on and one that cannot, so neither is ever suppressed.
   */
  error(message: string, error?: unknown, visible = false): void {
    const detail = error === undefined ? '' : `: ${describeError(error)}`;
    const fullMessage = `${message}${detail}`;
    this.write('ERROR', fullMessage);
    const stack = errorStack(error);
    if (stack) {
      for (const line of stack.split(/\r?\n/).slice(1)) {
        if (line.trim()) {
          this.write('ERROR', `    ${line.trim()}`);
        }
      }
    }
    if (visible) {
      void vscode.window.showErrorMessage(`Local LLM: ${fullMessage}`, 'Show Logs').then((action) => {
        if (action === 'Show Logs') {
          this.show();
        }
      });
    }
  }

  info(message: string): void {
    if (this.level !== 'error') {
      this.write('INFO', message);
    }
  }

  warn(message: string, visible = false): void {
    this.write('WARN', message);
    if (visible) {
      void vscode.window.showWarningMessage(`Local LLM: ${message}`, 'Show Logs').then((action) => {
        if (action === 'Show Logs') {
          this.show();
        }
      });
    }
  }

  debug(message: string): void {
    if (this.level === 'debug') {
      this.write('DEBUG', message);
    }
  }

  show(): void {
    this.channel.show(true);
  }

  dispose(): void {
    this.channel.dispose();
  }

  private write(kind: string, message: string): void {
    this.channel.appendLine(`[${new Date().toISOString()}] [${kind}] ${message}`);
  }
}
