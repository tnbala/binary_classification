import pickle 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
with open(r"D:\DA_and_DS_class\Data_science\ds_project2\Final_proj_docs\model.pkl","rb") as m:
    model = pickle.load(m)
with open(r"D:\DA_and_DS_class\Data_science\ds_project2\Final_proj_docs\dep_encoder.pkl","rb") as m:
    dep = pickle.load(m)
with open(r"D:\DA_and_DS_class\Data_science\ds_project2\Final_proj_docs\reg_encoder.pkl","rb") as m:
    reg = pickle.load(m)
with open(r"D:\DA_and_DS_class\Data_science\ds_project2\Final_proj_docs\edu_encoder.pkl","rb") as m:
    edu = pickle.load(m)
with open(r"D:\DA_and_DS_class\Data_science\ds_project2\Final_proj_docs\gen_encoder.pkl","rb") as m:
    gen = pickle.load(m)
with open(r"D:\DA_and_DS_class\Data_science\ds_project2\Final_proj_docs\rec_encoder.pkl","rb") as m:
    rec = pickle.load(m)
def promoted(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11):  
    x1= dep.transform(np.array([x1]))
    x2= reg.transform(np.array([x2]))
    x3= edu.transform(np.array([x3]))
    x4= gen.transform(np.array([x4]))
    x5= rec.transform(np.array([x5]))
    x12 = (x7+x9)/2
    return model.predict(np.array([[x1,x2,x3,x4,x5,x6,x8,x10,x11,x12]]))[0]