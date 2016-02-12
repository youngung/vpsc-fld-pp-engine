def count_paths(FLDA_tar_fn):
    """
    count number of paths in the tar file

    Arguments
    ---------
    FLDA_tar_fn
    """
    import os

    cmd = 'tar -tf %s'%FLDA_tar_fn
    p = os.popen(cmd)
    archived_files = p.read().split('\n')
    n_probe = 0
    for i in xrange(len(archived_files)):
        try:
            _fn_ = archived_files[i].split('.tar')[0]
            _ind_ = int(_fn_[::-1][:2])
        except:
            pass
        else:
            n_probe=n_probe+1
            # print _fn_
    return n_probe
