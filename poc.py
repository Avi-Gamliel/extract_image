import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def vertical_projection(binary):
    vertical_projection = np.sum(binary, axis=0)
    return vertical_projection

def horizontal_projection(binary):
    horizontal_projection = np.sum(binary, axis=1)
    return horizontal_projection

def image_to_binary(image, thresh=130):
    _, binary_image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def read_image(image):
    return cv2.imread(image,cv2.IMREAD_GRAYSCALE)



def recorsive_image(image):
    '''
    split the image from horizontal
    and count the space between the rows
    like 2, 2, 2, 20, 2, 2, 2, 20 , 2, 2, 2, ,2
    :return:
    '''
    binary_image = image_to_binary(image)
    # show_image(binary_image)

    # split blocks
    blocks =horizontal_projection(binary_image)
    bbox_blocks = []
    start_block = None
    end_block = None
    for index_block, block in enumerate(blocks):
        if block == 0:
            if index_block > 0:
                if blocks[index_block-1] > 0:
                    if bbox_blocks[index_block + 1]["start"] - bbox_block["end"] > bbox_block["end"] - bbox_block[
                        "start"]:
                        bbox_blocks.append({
                            "start":start_block,
                            "end":index_block
                        })
                    end_row = index_block
                if index_block < len(blocks)-1:
                    if blocks[index_block + 1] >0:
                        start_block  = index_block

    final_bbox_blocks= []
    start_block = None

    # fixed to split paragraph
    for index_bbox_block,bbox_block in enumerate(bbox_blocks):
        # print("height row",x["end"] - x["start"] )
        # print(bbox[i+1]["start"] - x["end"] )
        if start_block == None :
            start_block = bbox_block['start']
        if index_bbox_block +1 < len(bbox_blocks):
            if bbox_blocks[index_bbox_block+1]["start"] - bbox_block["end"] > bbox_block["end"] - bbox_block["start"]:
                final_bbox_blocks.append({"start":start, "end":bbox_block["end"]})
                start = bbox_blocks[index_bbox_block+1]["start"]
        else:
            final_bbox_blocks.append({"start": start, "end": bbox_block["end"]})

    final_bbox_column = []

    bbox_columns = []

    for bbox in final_bbox_blocks :

        print('----')
        crop = image[bbox["start"]:bbox["start"]+(bbox["end"]- bbox["start"]), 0:image.shape[1]]
        # show_image(crop)
        # check columns
        binary_crop = image_to_binary(crop)
        columns = vertical_projection(binary_crop)
        start_column = None
        # print('===>',)
        for index_column, column in enumerate(columns):
            if column == 0:
                if index_column > 0:
                    if columns[index_column - 1] > 0:
                        bbox_columns.append({
                            "start": start_column,
                            "end": index_column
                        })
                        end_column = index_column
                    if index_column < len(columns) - 1:
                        if columns[index_column + 1] > 0:
                            start_column = index_column
        final_bbox_columns = []
        start_c = None
        for j,  y in enumerate(bbox_columns):
            # print("height row",x["end"] - x["start"] )
            # print(bbox[i+1]["start"] - x["end"] )
            if start_c == None:
                start_c = y['start']
            # print(bbox_columns)
            if j + 1 < len(bbox_columns):
                if bbox_columns[j + 1]["start"] - y["end"] > 30:
                    final_bbox_columns.append({"start": start_c, "end": y["end"]})
                    start_c = bbox_columns[j + 1]["start"]
            else:
                final_bbox.append({"start": start_c, "end": y["end"]})


        print("final_bbox_columns", final_bbox_columns)
        # print(x["end"] - x["start"])
    print(horizontal_projection(binary_image))


# image = read_image('test.png')
# recorsive_image(image)
