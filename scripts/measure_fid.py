from cleanfid import fid


FDIR1 = '../results_0'
GT = '../gt'



def measure_fid(fdir1, fdir2):
    score = fid.compute_fid(fdir1, fdir2)
    print(FDIR1, ' : FID SCORE : ', score)


if __name__ == '__main__':
    measure_fid(FDIR1, GT)
