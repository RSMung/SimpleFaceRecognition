
import os

gpu_id = 0
# gpu_id = 1
# gpu_id = 2
# gpu_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


from CosFace.train_cosface import trainCosFaceMain
from ArcFace.train_arcface import trainArcFaceMain
from ElasticFace.train_elasticface import trainElasticFaceMain
from Softmax.train_softmax import trainSoftmaxMain
from UniFace.train_uniface import trainUniFaceMain
from TripletLoss.train_triplet_loss import trainTripletLossMain


flag1 = "trainArcFaceMain"
flag2 = "trainSoftmaxMain"
flag3 = "trainElasticFaceMain"
flag4 = "trainUniFaceMain"
flag5 = "trainTripletLossMain"
flag6 = "trainCosFaceMain"





# flag = flag1   # arcface
# flag = flag2   # softmax
# flag = flag3   # elasticface
# flag = flag4   # uniface
# flag = flag5   # trainTripletLossMain
flag = flag6   # cosface



if flag == flag1:
    trainArcFaceMain()
elif flag == flag2:
    trainSoftmaxMain()
elif flag == flag3:
    trainElasticFaceMain()
elif flag == flag4:
    trainUniFaceMain()
elif flag == flag5:
    trainTripletLossMain()
elif flag == flag6:
    trainCosFaceMain()
else:
    raise RuntimeError(f"flag:{flag} is invalid!")