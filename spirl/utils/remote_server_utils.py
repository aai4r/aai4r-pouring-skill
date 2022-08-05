import ftplib
import os


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