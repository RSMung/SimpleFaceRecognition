
import os
# gpu_id = 0
# gpu_id = 1
# gpu_id = 2
gpu_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


from ArcFace.train_arcface import trainArcFaceMain
from ElasticFace.train_elasticface import trainElasticFaceMain
from Softmax.train_softmax import trainSoftmaxMain
from UniFace.train_uniface import trainUniFaceMain


flag1 = "trainArcFaceMain"
flag2 = "trainSoftmaxMain"
flag3 = "trainElasticFaceMain"
flag4 = "trainUniFaceMain"





# flag = flag1
# flag = flag2
flag = flag3
# flag = flag4



if flag == flag1:
    trainArcFaceMain()
elif flag == flag2:
    trainSoftmaxMain()
elif flag == flag3:
    trainElasticFaceMain()
elif flag == flag4:
    trainUniFaceMain()
else:
    raise RuntimeError(f"flag:{flag} is invalid!")