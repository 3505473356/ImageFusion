import torch
import torch as t
from model_IR import Siamese, load_model, get_custom_CNN, jit_load, get_parametrized_model_IR
from model import Siamese, load_model, get_custom_CNN, jit_load, get_parametrized_model
from torchvision.transforms import Resize
from utils import get_shift, plot_samples, plot_displacement
import numpy as np
from scipy import interpolate
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
import cv2
import time
import sys

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")

# Set this according to your input
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
# MODEL_PATH = "./model_siam_eunord.pt"
MODEL1_PATH = "./model_IR.pt"
MODEL2_PATH = "./model_siam_eunord.pt"

# IR1_PATH = "./images/ir_1.png"
# IR2_PATH = "./images/ir_2.png"
# RGB1_PATH = "./images/rgb_1.png"
# RGB2_PATH = "./images/rgb_2.png"
# -------------------------------

WIDTH = 512  # - 8
PAD = 31
FRACTION = 8
OUTPUT_SIZE = WIDTH//FRACTION
CROP_SIZE = WIDTH - FRACTION
LAYER_POOL = False
FILTER_SIZE = 3
EMB_CHANNELS = 16
RESIDUALS = 0

size_frac = WIDTH / IMAGE_WIDTH
transform = Resize(int(IMAGE_HEIGHT * size_frac))
fraction_resized = int(FRACTION/size_frac)

def evaluate(histogram):
    shift_hist = histogram.cpu()
    f = interpolate.interp1d(np.linspace(0, IMAGE_WIDTH, OUTPUT_SIZE), shift_hist, kind="cubic")
    interpolated = f(np.arange(IMAGE_WIDTH))
    ret = -(np.argmax(interpolated) - (IMAGE_WIDTH) // 2.0)
    return histogram, ret

def run(model1, model2, IR1_PATH, IR2_PATH, RGB1_PATH, RGB2_PATH):
    
    if IR1_PATH == IR2_PATH or RGB1_PATH == RGB2_PATH or IR1_PATH == RGB1_PATH or IR2_PATH == RGB2_PATH or IR1_PATH == RGB2_PATH or IR2_PATH == RGB1_PATH:
        print("Wrong address")
        sys.exit()

    source_IR, target_IR = transform(read_image(IR1_PATH) / 255.0).to(device),\
                         transform(read_image(IR2_PATH) / 255.0).to(device)[..., FRACTION//2:-FRACTION//2]
        
    source_RGB, target_RGB = transform(read_image(RGB1_PATH) / 255.0).to(device),\
                         transform(read_image(RGB2_PATH) / 255.0).to(device)[..., FRACTION//2:-FRACTION//2]

    model1.eval()
    start = time.time()
    with torch.no_grad():
        source_ir_rep = model1.get_repr_ir(source_IR.unsqueeze(0))
        targert_ir_rep = model1.get_repr_ir(target_IR.unsqueeze(0))

        source_rgb_rep = model1.get_repr_rgb(source_RGB.unsqueeze(0))
        targert_rgb_rep = model1.get_repr_rgb(target_RGB.unsqueeze(0))

        histogram_ir = model1.match_corr(source_ir_rep, targert_ir_rep, padding=32)
        histogram_ir = model1.out_batchnorm(histogram_ir)
        histogram_ir = histogram_ir.squeeze(1).squeeze(1)
        histogram_ir = (histogram_ir - t.mean(histogram_ir)) / t.std(histogram_ir)
        histogram_ir = t.softmax(histogram_ir, dim=1)

        histogram_rgb = model1.match_corr(source_rgb_rep, targert_rgb_rep, padding=32)
        histogram_rgb = model1.out_batchnorm(histogram_rgb)
        histogram_rgb = histogram_rgb.squeeze(1).squeeze(1)
        histogram_rgb = (histogram_rgb - t.mean(histogram_rgb)) / t.std(histogram_rgb)
        histogram_rgb = t.softmax(histogram_rgb, dim=1)

        histogram_IR = histogram_ir * histogram_rgb
        end1 = time.time()

        histogram_1 = model1(source_RGB.unsqueeze(0), target_IR.unsqueeze(0), padding=PAD)
        histogram_1_co = histogram_1
        histogram_1 = (histogram_1 - t.mean(histogram_1)) / t.std(histogram_1)
        histogram_1 = t.softmax(histogram_1, dim=1)
        
        histogram_2 = model1(source_IR.unsqueeze(0), target_RGB.unsqueeze(0), padding=PAD)
        histogram_2_co = histogram_2
        histogram_2 = (histogram_2 - t.mean(histogram_2)) / t.std(histogram_2)
        histogram_2 = t.softmax(histogram_2, dim=1)
        
        # histogram_fusion = histogram_IR_co * histogram_RGB_co
        # histogram_fusion = (histogram_fusion - t.mean(histogram_fusion)) / t.std(histogram_fusion)
        histogram_fusion = histogram_1 * histogram_2
        end2 = time.time()
        histogram_4fusion = histogram_1 * histogram_2 * histogram_IR
        end3 = time.time()
        # visualize:

        histogram_IR, err_IR = evaluate(histogram_IR)
        histogram_fusion, err_fusion = evaluate(histogram_fusion)
        histogram_4fusion, err_4fusion = evaluate(histogram_4fusion)
    end = time.time()
    t1, t2, t3= end1-start, end2-end1, end3-start
    print("run time for ir-ir", end1-start)
    print("run time for ir-rgb", end2-start)
    print("run time for four hists", end3-start)
    
    model2.eval()
    start = time.time()
    with torch.no_grad():
        histogram_RGB = model2(source_RGB.unsqueeze(0), target_RGB.unsqueeze(0), padding=PAD)
        histogram_RGB_co = histogram_RGB
        histogram_RGB = (histogram_RGB - t.mean(histogram_RGB)) / t.std(histogram_RGB)
        histogram_RGB = t.softmax(histogram_RGB, dim=1)
        end4 = time.time()
        
        histogram_RGB, err_RGB = evaluate(histogram_RGB)

        print("Estimated displacement (IR, RGB, fusion) is", str(err_IR), str(err_RGB), str(err_fusion), "pixels.")
        print("run time for single RGB", end4-start)
        t4 = end4-start

        return histogram_RGB, err_IR, err_RGB, err_fusion, err_4fusion, t1, t2, t3, t4

def load_dataset(ref_path, emb_path):
    used_imgs_number = 0

    ref_RGB_path = os.path.join(ref_path,"RGB")
    ref_IR_path = os.path.join(ref_path,"IR" )

    ref_RGB_list = os.listdir(ref_RGB_path)
    ref_RGB_list.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
    ref_IR_list = os.listdir(ref_IR_path)
    ref_IR_list.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))

    emb_RGB_path = os.path.join(emb_path,"RGB")
    emb_IR_path = os.path.join(emb_path,"IR" )
    emb_RGB_list = os.listdir(emb_RGB_path)
    emb_RGB_list.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
    emb_IR_list = os.listdir(emb_IR_path)
    emb_IR_list.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))

    # if len(ref_RGB_list) <= len(emb_RGB_list):
    #     used_imgs_number = len(ref_RGB_list)
    #     emb_RGB_list = emb_RGB_list[:used_imgs_number]
    #     emb_IR_list = emb_IR_list[:used_imgs_number]

    # else:
    #     used_imgs_number = len(emb_RGB_list)
    #     ref_RGB_list = ref_RGB_list[:used_imgs_number]
    #     ref_IR_list = ref_IR_list[:used_imgs_number]

    return ref_RGB_list, ref_IR_list, emb_RGB_list, emb_IR_list

if __name__ == '__main__':
    ret = os.path.exists("Mean IR errors.npy")
    if ret:
        Merrors_IR = np.load("Mean IR errors.npy")
        Merrors_RGB = np.load("Mean RGB errors.npy")
        Merrors_fusion = np.load("Mean fusion errors.npy")

    else:
        Merrors_IR = []
        Merrors_RGB = []
        Merrors_fusion = []
        Merrors_4fusion = []

    ret = os.path.exists("Standard deviation of IR errors.npy")
    if ret:
        std_IR = np.load("Standard deviation of IR errors.npy")
        std_RGB = np.load("Standard deviation of RGB errors.npy")
        std_fusion = np.load("Standard deviation of fusion errors.npy")

    else:
        std_IR = []
        std_RGB = []
        std_fusion = []
        std_4fusion = []


    #Load the whole dataset
    # path0_forward_path = "path0/forward/"
    # path0_back_path = "path0/back/"
    # path0_forward_list = os.listdir(path0_forward_path)
    # path0_back_list = os.listdir(path0_back_path)

    # path1_forward_path = "path1/forward/"
    # path1_back_path = "path1/back/"
    # path1_forward_list = os.listdir(path1_forward_path)
    # path1_back_list = os.listdir(path1_back_path)

    # path2_forward_path = "path2/forward/"
    # path2_back_path = "path2/back/"
    # path2_forward_list = os.listdir(path2_forward_path)
    # path2_back_list = os.listdir(path2_back_path)

    # test_forward_path = "/home/u2004/Desktop/ImgF/test/path0/forward/"
    # test_back_path = "/home/u2004/Desktop/ImgF/test/path0/back/"
    test_forward_path = "test/path1/forward/"
    # test_back_path = "test/test/path1/back/"
    test_back_path = "test/test/back/others"
    sunglare = "test/test/back/sunglare/"
    test_forward_list = os.listdir(test_forward_path)
    test_back_list = os.listdir(test_back_path)
    sunglare_list = os.listdir(sunglare)

    model1 = get_parametrized_model_IR(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS, RESIDUALS, PAD, device)
    model1 = load_model(model1, MODEL1_PATH)

    model2 = get_parametrized_model(LAYER_POOL, FILTER_SIZE, 256, RESIDUALS, PAD, device)
    model2 = load_model(model2, MODEL2_PATH)

    PATH = [test_forward_list, test_back_list]
    head = [test_forward_path, test_back_path]

    print(PATH[0])
    print(PATH[1])

    STRIDE_1 = 1
    END = 110

    name = []

    for j in range(1,2):
        print("The "+ str(j+1) + " cycle. ") #range(0,len(PATH[j])-1 )
        for i in range(0,len(sunglare_list) ):
            print(len(sunglare_list)) 
            for k in range(0, len(PATH[j])): #i+1, len(PATH[j])
                print(len(PATH[j]))
            # for j in range(len(PATH)):
            # print("******* " + str(PATH[j][i]) + " --- " + str(PATH[j][i+1]) + "/" + str(len(PATH)**2) + " started *******")
                errors_IR = []
                errors_RGB = []
                errors_fusion = []
                errors_4fusion = []
                T1 = []
                T2 = []
                T3 = []
                T4 = []
                # emb = str(str(PATH[j][i]).split('-')[1] + "-" + str(PATH[j][i]).split('-')[3] + "-" + str(PATH[j][i]).split('-')[4])
                emb = str(str(sunglare_list[i]).split('-')[1] + "-" + str(sunglare_list[i]).split('-')[3] + "-" + str(sunglare_list[i]).split('-')[4])
                ref = str(str(PATH[j][k]).split('-')[1] + "-" + str(PATH[j][k]).split('-')[3] + "-" + str(PATH[j][k]).split('-')[4])
                na = ref + " vs " + emb
                name.append(na)
                # ref_dataset_path = os.path.join(head[j], PATH[j][i])
                emb_dataset_path = os.path.join(sunglare, sunglare_list[i])
                ref_dataset_path = os.path.join(head[j], PATH[j][k])
                print(ref_dataset_path)
                print(emb_dataset_path)
                ref_RGB_list, ref_IR_list, emb_RGB_list, emb_IR_list = load_dataset(ref_dataset_path, emb_dataset_path)
                count = 0

                print((len(ref_RGB_list)))
                print((len(emb_RGB_list)))

                rat_ref_RGB = 1
                rat_ref_IR = 1
                rat_emb_RGB = 1
                rat_emb_IR = 1

                if len(ref_RGB_list) == len(emb_RGB_list):
                    END = 100

                else:
                    if len(ref_RGB_list) < 100 or len(emb_RGB_list) < 100:
                        END = 80
                
            # if len(ref_RGB_list) < len(emb_RGB_list):
            #     rat_ref_RGB = 1
            #     rat_ref_IR = 1
            #     rat_emb_RGB = len(emb_RGB_list)/len(ref_RGB_list)
            #     rat_emb_IR = len(emb_RGB_list)/len(ref_RGB_list)
            #     print(rat_emb_RGB)

            # else:
            #     rat_ref_RGB = float(len(emb_RGB_list)/len(ref_RGB_list))
            #     rat_ref_IR = float(len(emb_RGB_list)/len(ref_RGB_list))
            #     rat_emb_RGB = 1
            #     rat_emb_IR = 1
            #     print(rat_ref_RGB)

                for x in np.arange(0, END, STRIDE_1):
                    ref_RGB, ref_IR = os.path.join(ref_dataset_path, "RGB", ref_RGB_list[int(x * rat_ref_RGB)]), os.path.join(ref_dataset_path, "IR", ref_IR_list[int(x * rat_ref_IR)])
                    emb_RGB, emb_IR = os.path.join(emb_dataset_path, "RGB", emb_RGB_list[int(x * rat_emb_RGB)]), os.path.join(emb_dataset_path, "IR", emb_IR_list[int(x * rat_emb_IR)])

                    # cv2.imwrite("./bad_imgs/" + str(x) + "_vers_" + str(x) + str(PATH[j][k]) + "_" + str(count) + "pairs" + "_IR_ref.png", cv2.imread(ref_IR))
                    # cv2.imwrite("./bad_imgs/" + str(x) + "_vers_" + str(x) + str(PATH[j][k]) + "_" + str(count) + "pairs" + "_IR_emb.png", cv2.imread(emb_IR))

                    histogram_1, err_IR, err_RGB, err_fusion, err_4fusion, t1, t2, t3, t4 = run(model1, model2, ref_IR, emb_IR, ref_RGB, emb_RGB)
                    T1.append(t1)
                    T2.append(t2)
                    T3.append(t3)
                    T4.append(t4)
                    irir = np.mean(T1)
                    irrgb = np.mean(T2)
                    four = np.mean(T3)
                    rgb = np.mean(T4)


                    # f, ax = plt.subplots(1,1, figsize=(6,4.5))
                    # aaa = np.arange(-len(histogram_1.numpy().squeeze())/2, len(histogram_1.numpy().squeeze())/2, dtype=int)
                    # print(aaa)
                    # print(histogram_1.numpy().squeeze())
                    # ax.bar(aaa, histogram_1.numpy().squeeze(), color='c')
                    # ax.set_title("likelihood histogram", fontsize=14)
                    # ax.set_xlabel("displacement", fontsize=14)
                    # ax.set_ylabel("likelihood", fontsize=14)
                    # plt.savefig("chatu.png")
                    # plt.close()

                    errors_IR.append(err_IR)
                    errors_RGB.append(err_RGB)
                    errors_fusion.append(err_fusion)
                    errors_4fusion.append(err_4fusion)

                    if abs(err_IR) >= 16:
                        exist = os.path.exists("./bad_imgs/" + str(PATH[j][i]) + "_vers_" + str(PATH[j][k]))
                        if not exist:
                            os.mkdir("./bad_imgs/" + str(PATH[j][i]) + "_vers_" + str(PATH[j][k]))

                        cv2.imwrite("./bad_imgs/" + str(PATH[j][i]) + "_vers_" + str(PATH[j][k]) + "/" + str(PATH[j][k]) + "_" + str(count) + "pairs" + str(int(x * rat_ref_IR)) + "_IR_ref.png", cv2.imread(ref_IR))
                        cv2.imwrite("./bad_imgs/" + str(PATH[j][i]) + "_vers_" + str(PATH[j][k]) + "/" + str(PATH[j][k]) + "_" + str(count) + "pairs" + str(int(x * rat_emb_IR)) + "_IR_emb.png", cv2.imread(emb_IR))

                        print(ref_IR_list[int(x * rat_ref_IR)])
                        print(emb_IR_list[int(x * rat_emb_IR)])
                    if abs(err_RGB) >= 16:
                        exist = os.path.exists("./bad_imgs/" + str(PATH[j][i]) + "_vers_" + str(PATH[j][k]))
                        if not exist:
                            os.mkdir("./bad_imgs/" + str(PATH[j][i]) + "_vers_" + str(PATH[j][k]))
                            
                        cv2.imwrite("./bad_imgs/" + str(PATH[j][i]) + "_vers_" + str(PATH[j][k]) + "/" + str(PATH[j][k]) + "_" + str(count) + "pairs" + str(int(x * rat_ref_RGB)) + "_RGB_ref.png", cv2.imread(ref_RGB))
                        cv2.imwrite("./bad_imgs/" + str(PATH[j][i]) + "_vers_" + str(PATH[j][k]) + "/" + str(PATH[j][k]) + "_" + str(count) + "pairs" + str(int(x * rat_emb_RGB)) + "_RGB_emb.png", cv2.imread(emb_RGB))
                        print(ref_RGB_list[int(x * rat_ref_RGB)])
                        print(emb_RGB_list[int(x * rat_emb_RGB)])

                    count += 1
                    print(str(x * rat_ref_RGB) + " / " + str(len(ref_RGB_list)))
            
                errors_IR = np.array(errors_IR).reshape(-1)
                errors_RGB = np.array(errors_RGB).reshape(-1)
                errors_fusion = np.array(errors_fusion).reshape(-1)
                errors_4fusion = np.array(errors_4fusion).reshape(-1)

                print(errors_fusion.shape)
                print(type(errors_fusion))
                print(np.min(np.abs(errors_IR)))

                with open('result.txt', 'a') as f:
                    f.write("************************ " + str(time.ctime()) + " start " + str(PATH[j][i]) + " ************************ " + '\r\n')
                    f.write("Fusion_non_cross: mean, " + str(np.mean(np.abs(errors_IR))) + " max, " + str(np.max(errors_IR)) + " min, " + str(np.min(errors_IR)) + " best, " + str(np.min(np.abs(errors_IR))) + " std, " + str(np.std(errors_IR)) +'\r\n')
                    f.write("RGB:  mean, " + str(np.mean(np.abs(errors_RGB))) + " max, " + str(np.max(errors_RGB)) + " min, " + str(np.min(errors_RGB)) + " best, " + str(np.min(np.abs(errors_RGB))) + " std, " + str(np.std(errors_RGB)) + '\r\n')
                    f.write("Fusion_cross:  mean, " + str(np.mean(np.abs(errors_fusion))) + " max, " + str(np.max(errors_fusion)) + " min, " + str(np.min(errors_fusion)) + " best, " + str(np.min(np.abs(errors_fusion))) + " std, " + str(np.std(errors_fusion)) + '\r\n')
                    f.write("Fusion_4:  mean, " + str(np.mean(np.abs(errors_4fusion))) + " max, " + str(np.max(errors_4fusion)) + " min, " + str(np.min(errors_4fusion)) + " best, " + str(np.min(np.abs(errors_4fusion))) + " std, " + str(np.std(errors_fusion)) + '\r\n')
                    f.write("************************ Runing Time: irir: " + str(irir) + ",irrgb" + str(irrgb) + " ,four" + str(four)+" ,rgb "+ str(rgb) + "  ************************ " + '\r\n')
                    f.write("************************ " + str(PATH[j][k]) + " finished ************************ " + '\r\n')
                    f.write("                                                                                           " + '\r\n')
                    f.close()
                
                print("Fusion_non_cross: mean, " + str(np.mean(np.abs(errors_IR))) + " max, " + str(np.max(errors_IR)) + " min, " + str(np.min(errors_IR)) + " best, " + str(np.min(np.abs(errors_IR))) + " std, " + str(np.std(errors_IR)))
                print("RGB:  mean, " + str(np.mean(np.abs(errors_RGB))) + " max, " + str(np.max(errors_RGB)) + " min, " + str(np.min(errors_RGB)) + " best, " + str(np.min(np.abs(errors_RGB))) + " std, " + str(np.std(errors_RGB)))
                print("Fusion_cross:  mean, " + str(np.mean(np.abs(errors_fusion))) + " max, " + str(np.max(errors_fusion)) + " min, " + str(np.min(errors_fusion)) + " best, " + str(np.min(np.abs(errors_fusion))) + " std, " + str(np.std(errors_fusion)))

                if np.mean(np.abs(errors_IR)) - np.mean(np.abs(errors_RGB)) < -1:

                    with open('IR_better.txt', 'a') as f:
                        f.write("************************ " + str(time.ctime()) + " start " + str(PATH[j][i]) + " ************************ " + '\r\n')
                        f.write("IR: mean, " + str(np.mean(np.abs(errors_IR))) + " max, " + str(np.max(errors_IR)) + " min, " + str(np.min(errors_IR)) + " best, " + str(np.min(np.abs(errors_IR))) + " std, " + str(np.std(errors_IR)) +'\r\n')
                        f.write("RGB:  mean, " + str(np.mean(np.abs(errors_RGB))) + " max, " + str(np.max(errors_RGB)) + " min, " + str(np.min(errors_RGB)) + " best, " + str(np.min(np.abs(errors_RGB))) + " std, " + str(np.std(errors_RGB)) + '\r\n')
                        f.write("Fusion:  mean, " + str(np.mean(np.abs(errors_fusion))) + " max, " + str(np.max(errors_fusion)) + " min, " + str(np.min(errors_fusion)) + " best, " + str(np.min(np.abs(errors_fusion))) + " std, " + str(np.std(errors_fusion)) + '\r\n')
                        f.write("************************ " + str(PATH[j][k]) + " finished ************************ " + '\r\n')
                        f.write("                                                                                           " + '\r\n')
                        f.close()

                if np.mean(np.abs(errors_4fusion)) - np.mean(np.abs(errors_RGB)) < -1:

                    with open('fusion_better.txt', 'a') as f:
                        f.write("************************ " + str(time.ctime()) + " start " + str(PATH[j][i]) + " ************************ " + '\r\n')
                        f.write("IR: mean, " + str(np.mean(np.abs(errors_IR))) + " max, " + str(np.max(errors_IR)) + " min, " + str(np.min(errors_IR)) + " best, " + str(np.min(np.abs(errors_IR))) + " std, " + str(np.std(errors_IR)) +'\r\n')
                        f.write("RGB:  mean, " + str(np.mean(np.abs(errors_RGB))) + " max, " + str(np.max(errors_RGB)) + " min, " + str(np.min(errors_RGB)) + " best, " + str(np.min(np.abs(errors_RGB))) + " std, " + str(np.std(errors_RGB)) + '\r\n')
                        f.write("Fusion:  mean, " + str(np.mean(np.abs(errors_fusion))) + " max, " + str(np.max(errors_fusion)) + " min, " + str(np.min(errors_fusion)) + " best, " + str(np.min(np.abs(errors_fusion))) + " std, " + str(np.std(errors_fusion)) + '\r\n')
                        f.write("************************ " + str(PATH[j][k]) + " finished ************************ " + '\r\n')
                        f.write("                                                                                           " + '\r\n')
                        f.close()
                
                f, ax = plt.subplots(4,1, figsize=(8,12))
                ax[0].plot(errors_IR, label = 'fusion_non_cross_errors')
                ax[0].set_title("fusion_non_cross_errors")
                ax[1].plot(errors_RGB, label = 'RGB_errors')
                ax[1].set_title("RGB_errors")
                ax[2].plot(errors_fusion, label = 'fusion_cross_errors')
                ax[2].set_title("fusion_cross_errors")
                ax[3].plot(errors_4fusion, label = 'fusion_4errors')
                ax[3].set_title("fusion_4errors")
                plt.savefig(str(PATH[j][i]) + "_vers_" + str(PATH[j][k]) + ".png")
                plt.close()

                np.save(str(PATH[j][i]) + "_vers_" + str(PATH[j][k]) + "_IR.npy", errors_IR)
                np.save(str(PATH[j][i]) + "_vers_" + str(PATH[j][k]) + "_RGB.npy", errors_RGB)
                np.save(str(PATH[j][i]) + "_vers_" + str(PATH[j][k]) + "_fusion.npy", errors_fusion)

                print("************************ " + str(PATH[j][i]) + "_vers_" + str(PATH[j][k]) + " finished ************************ ")
                print("                                                                                   ")
                
                Merrors_IR = np.append(Merrors_IR, np.mean(np.abs(errors_IR)))
                Merrors_RGB = np.append(Merrors_RGB, np.mean(np.abs(errors_RGB)))
                Merrors_fusion = np.append(Merrors_fusion, np.mean(np.abs(errors_fusion)))
                Merrors_4fusion = np.append(Merrors_4fusion, np.mean(np.abs(errors_4fusion)))
                std_IR = np.append(std_IR, np.std(errors_IR))
                std_RGB = np.append(std_RGB, np.std(errors_RGB))
                std_fusion = np.append(std_fusion, np.std(errors_fusion))
                std_4fusion = np.append(std_4fusion, np.std(errors_4fusion))
                np.save("Mean IR-IR errors.npy", Merrors_IR)
                np.save("Mean RGB errors.npy", Merrors_RGB)
                np.save("Mean IR-RGB errors.npy", Merrors_fusion)
                np.save("Mean four errors.npy", Merrors_4fusion)
                np.save("Standard deviation of IR-IR errors.npy", std_IR)
                np.save("Standard deviation of RGB errors.npy", std_RGB)
                np.save("Standard deviation of IR-RGB errors.npy", std_fusion)
                np.save("Standard deviation of 4fusion errors.npy", std_4fusion)
                np.save("name.npy", name)

        f, ax = plt.subplots(2,1, figsize=(20,16))
        x = np.arange(0, len(name))
        x_major_locator = plt.MultipleLocator(2)
        ax[0].plot(x, Merrors_IR, '.-', label = 'IR-IR, RGB-RGB: ' + str(round(np.mean(Merrors_IR), 2)), color='g')
        ax[0].plot(x, Merrors_RGB, '.-', label = 'RGB: ' + str(round(np.mean(Merrors_RGB), 2)), color='r')
        ax[0].plot(x, Merrors_fusion, '.-', label = 'IR-RGB: ' + str(round(np.mean(Merrors_fusion), 2)), color='b')
        ax[0].plot(x, Merrors_4fusion, '.-', label = 'IR-IR, RGB-RGB, IR-RGB: ' + str(round(np.mean(Merrors_4fusion), 2)), color='c')
        ax[0].legend()
        ax[0].set_title("Mean AE of methods")
        # ax[0].xaxis.set_major_locator(x_major_locator)
        # ax[0].set_xlabel("Comparison time")
        # ax[0].set_xticklabels(name, rotation=30)
        ax[0].set_ylabel("Mean AE")

        # ax[1].plot(Merrors_IR, '.-', label = 'IR-IR, RGB-RGB', color='g')
        # ax[1].plot(Merrors_fusion, '.-', label = 'IR-RGB', color='b')
        # ax[1].plot(Merrors_4fusion, '.-', label = 'IR-RGB, IR-IR, RGB-RGB', color='b')
        # # ax[1].legend()
        # # ax[1].set_title("Errors")
        # # ax[1].set_xlabel("pair rosbag series")
        # ax[1].set_xticklabels(name, rotation=30)
        # ax[1].set_ylabel("Mean errors")

        ax[1].plot(x, std_IR, '.-', label = 'IR-IR, RGB-RGB: ' + str(round(np.mean(std_IR), 2)), color='g')
        ax[1].plot(x, std_RGB, '.-', label = 'RGB: ' + str(round(np.mean(std_RGB), 2)), color='r')
        ax[1].plot(x, std_fusion, '.-', label = 'IR-RGB: ' + str(round(np.mean(std_fusion), 2)), color='b')
        ax[1].plot(x, std_4fusion, '.-', label = 'IR-RGB, IR-IR, RGB-RGB: ' + str(round(np.mean(std_4fusion), 2)), color='c')
        # ax[1].xaxis.set_major_locator(x_major_locator)
        ax[1].set_title("Standard deviation of methods")
        ax[1].set_xlabel("Comparison time", labelpad=10)
        print(len(name))
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(name, rotation=-60)
        ax[1].set_ylabel("SD")
        ax[1].legend()
        # plt.show()
        plt.savefig("Result.png")
        plt.close()

# plt.hist(data['petal_length'],
# 		alpha=0.5, # the transaparency parameter
# 		label='petal_length')

# plt.hist(data['sepal_length'],
# 		alpha=0.5,
# 		label='sepal_length')

# plt.legend(loc='upper right')
# plt.title('Overlapping with both alpha=0.5')
# plt.show()
