import sys
sys.path.append('..')
from importmod import *

def encode_with_cutoff(arr, value, cutoff_value):

    encoded_arr = np.zeros_like(arr)  # 모든 요소를 0으로 초기화한 배열 생성

    # 일정 값 이후의 인덱스에 대해 one-hot encoding 수행
    indices = np.where(arr >= cutoff_value)
    encoded_arr[indices] = value

    return encoded_arr


def visualize_instance_segmentation(args, images_list,  masks_list, title=None, save=None):
    fig = plt.figure(figsize=(16,16))
    grid = ImageGrid(fig, 111, nrows_ncols=(4,4), axes_pad=0.08)

    for ax, feature, predict in zip(grid, images_list[:16], masks_list[:16]):
        feature = feature.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()

        feature = np.transpose(feature, (1,2,0))   
        # feature = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY) # (1, 128, 128)
        
        oneHot = np.zeros(args.image_resize, dtype=np.uint8)
        for idx in range(args.n_classes):
            predict[idx] = (predict[idx] - predict[idx].min()) / (predict[idx].max() - predict[idx].min()) * 255
            p = encode_with_cutoff(predict[idx], idx/args.n_classes*255, 200)
            p = p.astype(np.uint8)
            oneHot |= p
        
        
        ax.imshow(feature, cmap='gray', alpha=0.5)
        ax.imshow(oneHot, cmap='jet', alpha=0.5)
        ax.axis('off')                    
        if save:
            plt.savefig(save)

    # Display the title if specified.
    if title:
        print(title)
    # Save the figure if specified.

    # Show the plot.
    plt.show()
    plt.close()