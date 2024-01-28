import numpy as np
import torch
import torch.nn as nn

trees_backup = [
[1,12,35],
[2,12,35],
[3,12,35],
[4,6,9],
[5,4,4],
[6,4,4],
[7,4,4],
[8,4,4],
[9,8,18],
[10,8,18],
[11,8,18],
[12,8,18],
[13,8,18],
[14,8,13],
[15,8,13],
[16,8,13],
[17,8,13],
[18,8,26],
[19,8,21],
[20,8,19],
[21,8,24],
[22,3,3],
[23,13,37],
[24,13,37],
[25,13,37],
[26,8,18],
[27,8,18],
[28,8,14],
[29,8,15],
[30,8,15],
[31,6,9],
[32,6,9],
[33,6,9],
[34,8,16],
[35,8,16],
[36,10,33],
[37,8,30],
[38,8,30],
[39,8,30],
[40,8,30],
[41,8,30],
[42,8,30],
[43,8,30],
[44,13,38],
[45,12,36],
[46,1,1],
[47,8,16],
[48,8,16],
[49,8,18],
[50,11,34],
[51,11,34],
[52,11,34],
[53,11,34],
[54,8,13],
[55,8,16],
[56,8,16],
[57,8,13],
[58,4,4],
[59,4,5],
[60,4,5],
[61,4,5],
[62,4,5],
[63,4,5],
[64,4,5],
[65,4,5],
[66,4,5],
[67,2,2],
[68,2,2],
[69,2,2],
[70,2,2],
[71,4,6],
[72,4,6],
[73,8,15],
[74,8,15],
[75,8,15],
[76,8,24],
[77,8,30],
[78,8,30],
[79,5,7],
[80,5,7],
[81,5,7],
[82,5,7],
[83,5,7],
[84,5,8],
[85,8,11],
[86,7,10],
[87,1,1],
[88,8,18],
[89,1,1],
[90,1,1],
[91,8,21],
[92,3,3],
[93,8,15],
[94,8,27],
[95,8,18],
[96,8,18],
[97,8,18],
[98,8,18],
[99,8,23],
[100,9,32],
[101,9,32],
[102,8,30],
[103,8,30],
[104,8,22],
[105,3,3],
[106,4,4],
[107,8,15],
[108,8,15],
[109,8,23],
[110,6,9],
[111,8,20],
[112,8,20],
[113,8,24],
[114,8,24],
[115,8,24],
[116,8,24],
[117,8,24],
[118,8,25],
[119,8,24],
[120,8,24],
[121,8,24],
[122,8,24],
[123,8,24],
[124,8,24],
[125,8,24],
[126,8,24],
[127,8,24],
[128,8,24],
[129,8,24],
[130,8,24],
[131,8,24],
[132,8,24],
[133,8,24],
[134,8,28],
[135,8,17],
[136,8,17],
[137,8,17],
[138,8,17],
[139,8,13],
[140,8,13],
[141,4,5],
[142,4,5],
[143,4,5],
[144,4,5],
[145,4,5],
[146,4,5],
[147,4,5],
[148,8,24],
[149,8,21],
[150,8,21],
[151,8,31],
[152,8,31],
[153,8,31],
[154,8,31],
[155,8,31],
[156,8,31],
[157,8,31],
[158,8,23],
[159,8,23],
[160,8,23],
[161,8,23],
[162,8,23],
[163,8,23],
[164,8,23],
[165,8,23],
[166,8,23],
[167,8,23],
[168,8,23],
[169,8,23],
[170,8,23],
[171,8,23],
[172,8,23],
[173,8,23],
[174,8,23],
[175,8,23],
[176,8,23],
[177,8,23],
[178,8,23],
[179,8,23],
[180,8,23],
[181,8,23],
[182,8,23],
[183,8,23],
[184,8,23],
[185,8,12],
[186,8,12],
[187,10,33],
[188,10,33],
[189,10,33],
[190,10,33],
[191,10,33],
[192,10,33],
[193,8,29],
[194,8,29],
[195,8,29],
[196,8,29],
[197,8,29],
[198,8,29],
[199,8,29],
[200,8,23]
]


trees = [
[1,212,35+213],
[2,212,35+213],
[3,212,35+213],
[4,206,9+213],
[5,204,4+213],
[6,204,4+213],
[7,204,4+213],
[8,204,4+213],
[9,208,18+213],
[10,208,18+213],
[11,208,18+213],
[12,208,18+213],
[13,208,18+213],
[14,208,13+213],
[15,208,13+213],
[16,208,13+213],
[17,208,13+213],
[18,208,26+213],
[19,208,21+213],
[20,208,19+213],
[21,208,24+213],
[22,203,3+213],
[23,213,37+213],
[24,213,37+213],
[25,213,37+213],
[26,208,18+213],
[27,208,18+213],
[28,208,14+213],
[29,208,15+213],
[30,208,15+213],
[31,206,9+213],
[32,206,9+213],
[33,206,9+213],
[34,208,16+213],
[35,208,16+213],
[36,210,33+213],
[37,208,30+213],
[38,208,30+213],
[39,208,30+213],
[40,208,30+213],
[41,208,30+213],
[42,208,30+213],
[43,208,30+213],
[44,213,38+213],
[45,212,36+213],
[46,201,1+213],
[47,208,16+213],
[48,208,16+213],
[49,208,18+213],
[50,211,34+213],
[51,211,34+213],
[52,211,34+213],
[53,211,34+213],
[54,208,13+213],
[55,208,16+213],
[56,208,16+213],
[57,208,13+213],
[58,204,4+213],
[59,204,5+213],
[60,204,5+213],
[61,204,5+213],
[62,204,5+213],
[63,204,5+213],
[64,204,5+213],
[65,204,5+213],
[66,204,5+213],
[67,202,2+213],
[68,202,2+213],
[69,202,2+213],
[70,202,2+213],
[71,204,6+213],
[72,204,6+213],
[73,208,15+213],
[74,208,15+213],
[75,208,15+213],
[76,208,24+213],
[77,208,30+213],
[78,208,30+213],
[79,205,7+213],
[80,205,7+213],
[81,205,7+213],
[82,205,7+213],
[83,205,7+213],
[84,205,8+213],
[85,208,11+213],
[86,207,10+213],
[87,201,1+213],
[88,208,18+213],
[89,201,1+213],
[90,201,1+213],
[91,208,21+213],
[92,203,3+213],
[93,208,15+213],
[94,208,27+213],
[95,208,18+213],
[96,208,18+213],
[97,208,18+213],
[98,208,18+213],
[99,208,23+213],
[100,209,32+213],
[101,209,32+213],
[102,208,30+213],
[103,208,30+213],
[104,208,22+213],
[105,203,3+213],
[106,204,4+213],
[107,208,15+213],
[108,208,15+213],
[109,208,23+213],
[110,206,9+213],
[111,208,20+213],
[112,208,20+213],
[113,208,24+213],
[114,208,24+213],
[115,208,24+213],
[116,208,24+213],
[117,208,24+213],
[118,208,25+213],
[119,208,24+213],
[120,208,24+213],
[121,208,24+213],
[122,208,24+213],
[123,208,24+213],
[124,208,24+213],
[125,208,24+213],
[126,208,24+213],
[127,208,24+213],
[128,208,24+213],
[129,208,24+213],
[130,208,24+213],
[131,208,24+213],
[132,208,24+213],
[133,208,24+213],
[134,208,28+213],
[135,208,17+213],
[136,208,17+213],
[137,208,17+213],
[138,208,17+213],
[139,208,13+213],
[140,208,13+213],
[141,204,5+213],
[142,204,5+213],
[143,204,5+213],
[144,204,5+213],
[145,204,5+213],
[146,204,5+213],
[147,204,5+213],
[148,208,24+213],
[149,208,21+213],
[150,208,21+213],
[151,208,31+213],
[152,208,31+213],
[153,208,31+213],
[154,208,31+213],
[155,208,31+213],
[156,208,31+213],
[157,208,31+213],
[158,208,23+213],
[159,208,23+213],
[160,208,23+213],
[161,208,23+213],
[162,208,23+213],
[163,208,23+213],
[164,208,23+213],
[165,208,23+213],
[166,208,23+213],
[167,208,23+213],
[168,208,23+213],
[169,208,23+213],
[170,208,23+213],
[171,208,23+213],
[172,208,23+213],
[173,208,23+213],
[174,208,23+213],
[175,208,23+213],
[176,208,23+213],
[177,208,23+213],
[178,208,23+213],
[179,208,23+213],
[180,208,23+213],
[181,208,23+213],
[182,208,23+213],
[183,208,23+213],
[184,208,23+213],
[185,208,12+213],
[186,208,12+213],
[187,210,33+213],
[188,210,33+213],
[189,210,33+213],
[190,210,33+213],
[191,210,33+213],
[192,210,33+213],
[193,208,29+213],
[194,208,29+213],
[195,208,29+213],
[196,208,29+213],
[197,208,29+213],
[198,208,29+213],
[199,208,29+213],
[200,208,23+213]
]


trees_order_to_family = [ 
[1,1],
[2,2],
[3,3],
[4,4,5,6],
[5,7,8],
[6,9],
[7,10],
[8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
[9,32],
[10,33],
[11,34],
[12,35,36],
[13,37,38],
]


trees_family_to_species = [ 
[1, 46, 87,89,90],
[2, 67, 68, 69, 70],
[3, 22, 92, 105],
[4, 5, 6, 7, 8, 58, 106],
[5, 59, 60, 61, 62, 63, 64, 65, 66, 141, 142, 143, 144, 145, 146, 147],
[6, 71,72],
[7, 79, 80, 81, 82, 83],
[8, 84],
[9, 4, 31, 32, 33, 110],
[10, 86],
[11, 85],
[12, 185, 186],
[13, 14,15,16,17, 54, 57, 139, 140],
[14, 28],
[15, 29, 30, 73, 74, 75, 93, 107, 108],
[16, 34, 35, 47, 48, 55, 56],
[17, 135, 136, 137, 138],
[18,9,10,11,12,13, 26, 27, 49, 88, 95, 96, 97, 98],
[19, 20],
[20, 111, 112],
[21, 19, 91, 149, 150],
[22, 104],
[23, 99, 109,  158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,\
 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 200],
[24, 21, 76, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131\
, 132, 133, 148],
[25, 118],
[26, 18],
[27, 94],
[28, 134],
[29, 193, 194, 195, 196, 197, 198, 199],
[30, 37, 38, 39, 40, 41, 42, 43, 77, 78, 102, 103],
[31, 151, 152, 153, 154, 155, 156, 157],
[32, 100, 101],
[33, 36, 187, 188, 189, 190, 191, 192],
[34, 50, 51, 52, 53],
[35, 1, 2, 3],
[36, 45],
[37, 23,24,25],
[38, 44],
]




trees_order_to_species = [
[1, 46, 87, 89, 90], 
[2, 67, 68, 69, 70], 
[3, 22, 92, 105],
[4, 5, 6, 7, 8, 58, 106, 59, 60, 61, 62, 63, 64, 65, 66, 141, 142, 143, 144, 145, 146, 147, 
71,72],  #1
[5, 79 , 80 , 81 , 82 , 83, 84],  #1
[6, 4, 31, 32, 33, 110],

[7, 86],
[8, 85, 
185, 186, 
14,15,16,17,54,57,139,140, 
28, 
29,30,73,74,75,93,107,108, 
34,35,47,48,55,56, 
135,136,137,138, 
9,10,11,12,13,26,27,49,88,95,96,97,98, 
20, 
111,112, 
19,91,149,150, 
104, 
99, 109, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 200, 
21, 76, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 148, 
118, 
18, 
94, 
 193, 194, 195, 196, 197, 198, 199, 
37, 38, 39, 40, 41, 42, 43, 77, 78, 102, 103,
151, 152, 153, 154, 155, 156, 157
],  #1
[9, 100, 101],
[10, 36, 187, 188, 189, 190, 191, 192],  #1
[11, 50, 51, 52, 53],
[12, 1,2,3,45],#1
[13, 23,24,25,44]
]

def get_order_family_target(targets):


    order_target_list = []
    family_target_list = []


    for i in range(targets.size(0)):

        order_target_list.append(trees[targets[i]][1]-1)
        family_target_list.append(trees[targets[i]][2]-1)



    order_target_list = Variable(torch.from_numpy(np.array(order_target_list)).cuda())   
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)).cuda())   

    return order_target_list, family_target_list

def get_label_list(targets):

    label_list = []
    # print(targets)
    for i in range(targets.size(0)):
        # print(targets[i].size())
        # print(targets[i])
        if trees[targets[i]][0] is not -1:
            # print(trees[targets[i]])
            # print('*****',targets[7])
            # print('-----',trees[targets[7]])
            # print(i, trees[targets[i]])
            last = trees[targets[i]].pop(0)
            trees[targets[i]].append(last)
            # print(trees[targets[i]])
            trees[targets[i]].insert(0, -1)
            trees[targets[i]].append(0)
            # print(trees[targets[i]])
            label_list.append([k+1 for k in trees[targets[i]]])
        else:
            label_list.append([k+1 for k in trees[targets[i]]])

    # print(label_list)
    dec_list = [d[:-1] for d in label_list]
    t_list = [t[1:] for t in label_list]

    # print(dec_list)
    dec_list = torch.from_numpy(np.array(dec_list)).cuda()
    t_list = torch.from_numpy(np.array(t_list)).cuda()
    return dec_list, t_list

if __name__ == '__main__':
    targets = torch.tensor([ 22,  15, 176, 134, 131, 185,  54, 116], device='cuda:0')
    k,t = get_label_list(targets)
    print(k)
    print(t)