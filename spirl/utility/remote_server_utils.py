import ftplib
import os
from spirl.utility.general_utils import AttrDict


def download_model_weights(params):
    """
        FTP server login
    """
    user = params.user
    pw = params.pw
    ip_addr = params.ip_addr

    ftp = ftplib.FTP(host=ip_addr)
    ftp.encoding = "utf-8"
    ftp.login(user=user, passwd=pw)
    print("ftp login success!")

    """
        Set task path
    """
    _path = "/home/tw_etri_server/project/cloudrobot/aai4r-pouring-skill/experiments/skill_prior_learning"
    task = "pouring_water_img"
    method = "hierarchical_cl"
    weight_type = "weights"
    path = os.path.join(_path, task, method, weight_type)

    # print("nlst: ", ftp.nlst(path))
    files = ftp.nlst(path)
    files = sorted(files, key=lambda x: int(x[x.find('_ep') + 3:x.find('.')]))
    print(files)
    target = files[-1]
    print("target: ", target)

    curr_dir = os.getcwd()
    os.chdir("./file_down")
    fd = open(target[target.rfind('/') + 1:], "wb")
    ftp.retrbinary("RETR %s" % target, fd.write)
    fd.close()
    print("complete..!")


"""
Folder Naming by network structure
A) w. pre_train(A1) / w.o. pre_train(A2)
B) w. robot_state(B1) / w.o. robot_state(B2)
C) w. aux_pred(C1) / w.o. aux_pred(C2)
D) w. recurrent prior(D1) / w.o. recurrent prior(D2)
E) and Next??

*** symbol
g: greater than (>)
l: less than (<)
ge: greater equal than (>=)
le: less equal than (<=)
"""


class WeightNaming:
    net_cond_list = {
        "use_pretrain": AttrDict(condition=True, true_tag="A1", false_tag="A2"),
        "state_cond": AttrDict(condition=True, true_tag="B1", false_tag="B2"),
        "aux_pred_dim": AttrDict(condition="g", to=0, true_tag="C1", false_tag="C2"),
        # g: greater(>), l: less(<), e: equal(=)
        "recurrent_prior": AttrDict(condition=True, true_tag="D1", false_tag="D2"),
    }

    def __init__(self):
        pass

    @classmethod
    def get_condition_result_from_str(cls, src, cmd, to):
        cmd_list = ["g", "l", "ge", "le"]
        if cmd not in cmd_list:
            raise KeyError("Unsupported command... Possible conditions are: {}".format(cmd_list))

        if cmd == "g":
            return src > to
        elif cmd == "l":
            return src < to
        elif cmd == "ge":
            return src >= to
        else:
            return src <= to

    @classmethod
    def weights_name_convert(cls, cfg):
        cfg.weights_dir += "_"
        for key, val in cls.net_cond_list.items():
            assert key in cfg
            if type(val.condition) is str:
                temp = cls.get_condition_result_from_str(src=cfg.get(key), cmd=val.condition, to=val.to)
                cfg.weights_dir += val.true_tag if temp else val.false_tag
                continue
            cfg.weights_dir += val.true_tag if cfg.get(key) == val.condition else val.false_tag

    @classmethod
    def print_condition_list(cls):
        print("Condition list: ", cls.net_cond_list)

    @classmethod
    def weights_name_dec(cls, weights_dir):
        assert weights_dir.find("_") > 0
        result = "weights__"
        for key, val in cls.net_cond_list.items():
            i = weights_dir.find(val.true_tag[0])
            check = "w." if int(weights_dir[i+1]) == 1 else "w.o."
            result += check + key + "__"
        return result[:-2]


# print(WeightNaming.print_condition_list())
# print("weights_name_decode:::: ")
# weights_dir = "weights_A1B1C2D1"
# print(WeightNaming.weights_name_dec(weights_dir))
