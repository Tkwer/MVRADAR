
import torch

def ride_augmentation(input_var1,input_var2,input_var3,input_var4,input_var5,v1=2,v2=2,v3=2,v4=1.5,v5=0.005):
    
    def shift_matrix2D_with_batch(input_tensor, x_offset, y_offset, enhance=1.2, noise_stddev=0.0001):
        # 将输入张量移动到 GPU 上
        
        batch_size, num_channels, rows, cols = input_tensor.shape

        # 创建一个新的张量，用于存储结果
        new_tensor = torch.zeros_like(input_tensor)

        # 计算均值张量，以便后续使用
        mean_tensor = torch.mean(input_tensor, dim=(2, 3), keepdim=True)

        # for batch in range(batch_size):
        #     for channels in range(num_channels):
                # 偏移操作
        new_i = torch.arange(rows).view(rows, 1).cuda() - y_offset
        new_j = torch.arange(cols).view(1, cols).cuda() - x_offset
        new_i = torch.clamp(new_i, 0, rows - 1).long()
        new_j = torch.clamp(new_j, 0, cols - 1).long()

        new_tensor= input_tensor[:,:,new_i,new_j]

        # 添加噪声和增强处理
        # zero_mean = torch.zeros(batch_size, num_channels, rows, cols, device='cuda')
        # noise_stddev = noise_stddev * torch.ones(batch_size, num_channels, rows, cols, device='cuda')
        noise = torch.normal(0.0, noise_stddev, size=new_tensor.shape).cuda()
        new_tensor = torch.clamp(new_tensor, min=1e-10)
        new_tensor = torch.clamp(new_tensor + noise, min=1e-10)
        new_tensor = (new_tensor ** enhance) / (new_tensor ** enhance + mean_tensor)
        return new_tensor

    def shift_matrix3D_with_batch(input_tensor, x_offset, y_offset, z_offset, enhance = 1.2, noise_stddev=0.0001):
        batch_size, num_frames, num_channels, rows, cols = input_tensor.shape
        # print(num_dimensions)
        mean_tensor = torch.mean(input_tensor, dim=(3, 4), keepdim=True)
        input_tensor = input_tensor.permute(0,2,1,3,4)
        new_tensor = torch.zeros_like(input_tensor,device='cuda')
        # 计算均值张量，以便后续使用
        

        # 计算偏移后的坐标
        new_d = torch.arange(num_frames).view(num_frames, 1, 1).cuda() - z_offset
        new_i = torch.arange(rows).view(1, rows, 1).cuda() - y_offset
        new_j = torch.arange(cols).view(1, 1, cols).cuda() - x_offset
        new_d = torch.clamp(new_d, 0, num_frames - 1).long()
        new_i = torch.clamp(new_i, 0, rows - 1).long()
        new_j = torch.clamp(new_j, 0, cols - 1).long()

        new_tensor = input_tensor[:,:,new_d,new_i,new_j]
        new_tensor = new_tensor.permute(0,2,1,3,4)
        # zero_mean = torch.zeros(batch_size, num_frames, num_channels, rows, cols, device='cuda')
        # noise_stddev = noise_stddev * torch.ones(batch_size, num_frames, num_channels, rows, cols, device='cuda')
        noise = torch.normal(0.0, noise_stddev, size=new_tensor.shape).cuda()
        new_tensor = torch.clamp(new_tensor + noise, min=1e-10)
        new_tensor = (new_tensor ** enhance) / (new_tensor ** enhance + mean_tensor)
        
        return new_tensor

    # time shifttorch.randint(start, stop, size=(1,)).item()
    offset1 = torch.randint(-v1, v1, size=(1,),device='cuda')
    # Range shift
    offset2 = torch.randint(-v2, v2, size=(1,),device='cuda')
    # Angle shift
    offset3 = torch.randint(-v3, v3, size=(1,),device='cuda')
    # en shift
    enhance = torch.rand(1,device='cuda') * (v4 - 0.8) + 0.8
    # offset1 = torch.ones(1,device='cuda')*v1
    # # Range shift
    # offset2 = torch.ones(1,device='cuda')*v2
    # # Angle shift
    # offset3 = torch.ones(1,device='cuda')*v3
    # # en shift
    # enhance = torch.ones(1,device='cuda')*v4
    noise_s = v5 
    zore_tensor = torch.zeros(1,device='cuda')

    DT_feature = shift_matrix2D_with_batch(input_var2, zore_tensor, offset1, enhance, noise_s)
    ART_feature = shift_matrix3D_with_batch(input_var1, offset2, offset3, offset1, enhance, noise_s)
    
    ERT_feature = shift_matrix3D_with_batch(input_var3, offset3, offset2, offset1, enhance, noise_s)
    RDT_feature = shift_matrix3D_with_batch(input_var4, zore_tensor, offset2, offset1, enhance, noise_s)
    RT_feature = shift_matrix2D_with_batch(input_var5, offset2, offset1*3, enhance, noise_s)


    return ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature