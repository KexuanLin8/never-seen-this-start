"""Read-only SO-101 bus check. Never writes registers or changes torque.

Run after supporting the arm and power-cycling its servo supply.
This is a stationary communication check, not a test under motor load.
"""

import argparse
import importlib.metadata
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    if args.samples < 1 or args.interval < 0.05:
        parser.error("samples must be >= 1 and interval must be >= 0.05 seconds")

    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    names = (
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    )
    motors = {
        name: Motor(index, "sts3215", MotorNormMode.RANGE_M100_100)
        for index, name in enumerate(names, 1)
    }
    print("READ ONLY: no goal writes, torque changes, calibration or PID changes.", flush=True)
    print("Port:", args.port, "| Python:", platform.python_version(), flush=True)
    for package in ("lerobot", "rustypot", "pyserial"):
        try:
            print(package + ":", importlib.metadata.version(package), flush=True)
        except importlib.metadata.PackageNotFoundError:
            print(package + ": not installed", flush=True)

    bus = SerialMotorsBus(port=args.port, motors=motors)
    failures = 0
    completed = 0
    consecutive_failures = 0
    durations = []
    try:
        # Opening the low-level bus bypasses Robot.connect/configure entirely.
        bus.connect(handshake=False)
        print("\nIndividual reads (raw register values):", flush=True)
        for name, motor in motors.items():
            try:
                position = bus.read("Present_Position", name, normalize=False, num_retry=0)
                print(f"ID {motor.id} {name}: position={position}", flush=True)
            except Exception as error:
                failures += 1
                print(f"ID {motor.id} {name}: POSITION FAIL: {error}", flush=True)
            for register in ("Torque_Enable", "Present_Voltage", "Present_Temperature"):
                try:
                    value = bus.read(register, name, normalize=False, num_retry=0)
                    print(f"  {register}={value} (raw)", flush=True)
                except Exception as error:
                    print(f"  {register}: unavailable: {error}", flush=True)

        print("\nRepeated sync reads; retries=0; arm stays stationary:", flush=True)
        for index in range(args.samples):
            started = time.perf_counter()
            try:
                positions = bus.sync_read("Present_Position", normalize=False, num_retry=0)
                elapsed = (time.perf_counter() - started) * 1000
                durations.append(elapsed)
                completed += 1
                consecutive_failures = 0
                values = ", ".join(f"{motors[name].id}:{value}" for name, value in positions.items())
                print(f"[{index + 1}/{args.samples}] OK {elapsed:.1f} ms | {values}", flush=True)
            except Exception as error:
                failures += 1
                consecutive_failures += 1
                print(f"[{index + 1}/{args.samples}] SYNC FAIL: {error}", flush=True)
                if consecutive_failures >= 3:
                    print("Stopping after 3 consecutive sync failures.", flush=True)
                    break
            time.sleep(args.interval)
    except Exception as error:
        failures += 1
        print(f"BUS FAIL: {error}", flush=True)
    finally:
        # The default disconnect would WRITE torque-off. Explicitly avoid that.
        bus.disconnect(disable_torque=False)

    print(f"\nSUMMARY: sync_ok={completed}; position/sync/bus_failures={failures}", flush=True)
    if durations:
        print(f"Successful sync latency: mean={sum(durations)/len(durations):.1f} ms; max={max(durations):.1f} ms", flush=True)
    print("A successful stationary test does NOT verify power stability under motor load.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
