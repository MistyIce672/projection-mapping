import sys
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AppReloader(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.start_app()

    def start_app(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        print("\nStarting application...")
        self.process = subprocess.Popen([sys.executable, 'main.py'])

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f"\nDetected change in {event.src_path}")
            self.start_app()

def main():
    reloader = AppReloader()
    observer = Observer()
    observer.schedule(reloader, path='.', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if reloader.process:
            reloader.process.terminate()
    observer.join()
    

if __name__ == "__main__":
    main() 