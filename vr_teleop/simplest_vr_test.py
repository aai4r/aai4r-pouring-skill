import triad_openvr
import sys
import time

from isaacgym import gymapi


def simple_controller_test():
    # TODO,, why...?
    gymapi.acquire_gym().create_sim(0, 0, gymapi.SIM_PHYSX, gymapi.SimParams())

    vr = triad_openvr.triad_openvr()
    vr.print_discovered_objects()

    if len(sys.argv) == 1:
        interval = 1 / 250
    elif len(sys.argv) == 2:
        interval = 1 / float(sys.argv[1])
    else:
        print("Invalid number of arguments")
        interval = False

    if interval:
        while True:
            start = time.time()
            pv, av = [], []
            for each in zip(vr.devices["controller_1"].get_velocity(), vr.devices["controller_1"].get_angular_velocity()):
                # [x, y, z, yaw, pitch, roll]
                pv.append(each[0]), av.append(each[1])
            print(pv + av)

            d = vr.devices["controller_1"].get_controller_inputs()
            if d['trigger']:
                print("trigger is pushed")

            sleep_time = interval - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    simple_controller_test()
