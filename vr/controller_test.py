import triad_openvr
import time
import sys

v = triad_openvr.triad_openvr()
v.print_discovered_objects()


def controller_test():
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
            txt = ""
            # for each in v.devices["controller_1"].get_pose_euler():
            for each in v.devices["controller_1"].get_velocity():
                # [x, y, z, yaw, pitch, roll]
                txt += "%.4f" % each
                txt += " "
            print("\r" + txt, end="")

            d = v.devices["controller_1"].get_controller_inputs()
            if d['trigger']:
                print("trigger is pushed")

            sleep_time = interval - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    controller_test()
