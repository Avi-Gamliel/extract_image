import numpy as np
import cv2
import fitz

from poc import image_to_binary


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def image_to_binary(image,thresh = 250):
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)
    return binary

def show(image, title="temp"):
    cv2.imshow(title,image)
    cv2.waitKey(0)

def horizontal_projection(image):
    return np.sum(image, axis=1)

def vertical_projection(image):
    return np.sum(image, axis=0)

def erode_image(image, y=1,x=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (x,y))
    bold_image = cv2.erode(image, kernel, iterations=1)
    return bold_image

def dilate_image(image, x=1,y=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x,y))
    bold_image = cv2.dilate(image, kernel, iterations=1)
    return bold_image

def blur_image(image, blur_size= 1):
    return cv2.blur(image, (blur_size, blur_size))

def split_blocks(blocks):
    global prev_block
    start_index = None
    block_bbox = []
    averge = []
    for index_block, block in enumerate(blocks):
        prev_block = None
        next_block = None
        if index_block > 0 and index_block < len(blocks) - 1:
            prev_block = blocks[index_block - 1]
            next_block = blocks[index_block + 1]

        if block == 0:
            if next_block or prev_block:

                if next_block > 0 and  prev_block ==0:
                    start_index = index_block+1

                if prev_block > 0:
                    if block_bbox:
                        averge.append(start_index - block_bbox[-1]["end"])
                    if start_index == None:
                        start_index = 0
                    block_bbox.append({
                        "start": start_index,
                        "end": index_block
                    })
                    start_index = index_block + 1
                    prev_block = 0

        else:
            if index_block == len(blocks) - 1:
                if start_index == None:
                    start_index = 0
                if len(blocks) > 0:
                    if len(block_bbox) >0:
                        if start_index != block_bbox[-1]["end"]:
                            if block_bbox:
                                averge.append(start_index - block_bbox[-1]["end"])
                            block_bbox.append({
                            "start": start_index,
                            "end": index_block
                            })
                    else:
                        if block_bbox:
                            averge.append(start_index - block_bbox[-1]["end"])
                        block_bbox.append({
                            "start": start_index,
                            "end": index_block
                        })
    # print(averge)
    # fixed block_bbox

    # space line
    start_index = None
    organize_block = []
    space_line = None
    # print(block_bbox)
    for index_bb, bb in enumerate(averge):
        if start_index == None:
            start_index = index_bb
        if space_line == None:
             space_line = bb
        elif space_line:
            if bb > space_line + 2:
                last_paragraph = index_bb-1
                end = block_bbox[last_paragraph]["end"]
                organize_block.append({
                    "start": block_bbox[start_index]["start"],
                    "end":end})
                start_index = None
    # print(organize_block)




    return block_bbox


def split_pdf_to_images(path):
    pdf = fitz.open(path)
    for index_page in range(len(pdf)):
        page = pdf[index_page]
        pix = page.get_pixmap()
        pix.save(f'page_{index_page}.png')


def clean_image(image):

    binary = image_to_binary(image)
    blocks_projection = vertical_projection(binary)
    # print(blocks_projection)
    start_index = None
    end_index = None

    for index_block, block in enumerate(blocks_projection):

        prev_block = None
        next_block = None

        if index_block > 0:
            prev_block = int(blocks_projection[index_block -1])
        if index_block+1 < len(blocks_projection):
            next_block = int(blocks_projection[index_block +1])


        if block == 0:
            if next_block:
                if next_block >0:
                    if start_index == None:
                        start_index = index_block + 1
            if prev_block:
                if prev_block >0:
                    end_index = index_block
        else:
            if index_block == 0:
                start_index = 0
            if index_block +1 == len(blocks_projection):
                end_index = index_block
        if index_block +1 == len(blocks_projection):
            if start_index == None:
                start_index = 0
            if end_index == None:
                end_index = image.shape[1]
    # print(start_index, end_index)
    crop_image = image[0:image.shape[0], start_index: start_index + (end_index - start_index)]
    return crop_image

def check_the_smaller_stroke(image):
    binary = image_to_binary(image)
    blocks_projection = vertical_projection(binary)

    start_index = None
    end_index = None
    stroke_width = None
    for index_block, block in enumerate(blocks_projection):

        prev_block = None
        next_block = None

        if index_block > 0:
            prev_block = int(blocks_projection[index_block - 1])
        if index_block + 1 < len(blocks_projection):
            next_block = int(blocks_projection[index_block + 1])

        if block == 0:
            if next_block:
                if next_block > 0:
                    if start_index == None:
                        start_index = index_block + 1
            if prev_block:
                if prev_block > 0:
                    end_index = index_block
        if index_block + 1 == len(blocks_projection):
            if start_index == None:
                start_index = 0
            if end_index == None:
                end_index = image.shape[1]
    if stroke_width:
        if image.shape[1] - end_index < stroke_width:
            stroke_width = image.shape[1] - end_index
        if start_index < stroke_width:
            stroke_width = start_index
    else:
        if start_index < image.shape[1] - end_index :
            stroke_width = start_index
        else:
            stroke_width = image.shape[1] - end_index

    return stroke_width
def check_border (image):
    binary = image_to_binary(image)
    _, invert_binary = cv2.threshold(binary, 0 ,255,cv2.THRESH_BINARY_INV)
    h_inv = horizontal_projection(invert_binary)
    v_inv = vertical_projection(invert_binary)
    stroke_width = check_the_smaller_stroke(binary)
    h = horizontal_projection(binary)
    v = vertical_projection(binary)
    height, width = image.shape
    max_h = width * 255
    max_v = height * 255

    if h[0]==max_h and  h[-1] == max_h and v[0] == max_v and v[-1] == max_v:
        # run h
        return {"status":True, "stroke_width":stroke_width}
    return {"status":False, "stroke_width":None}

def clean_crop_border(image):
    global x1, y1, x ,y2
    binary = image_to_binary(image)
    # show(binary)
    # _, invert_binary = cv2.threshold(binary, 0 ,255,cv2.THRESH_BINARY_INV)
    # h_inv = horizontal_projection(invert_binary)
    # v_inv = vertical_projection(invert_binary)
    h = horizontal_projection(binary)
    v = vertical_projection(binary)
    prev_x, prev_y, next_x, next_y = None, None,None,None
    height, width = image.shape
    max_h = width * 255
    max_v = height * 255

    x1, y1,x2,y2= 0,0,0,0
    # print(v)
    # print(h)
    for i, y in enumerate(h):
        if i > 0:
            prev_y = h[i - 1]
        if i +1 < len(h):
            next_y = h[i +1]

        if y == max_v:
            if next_y != None:
                if y1 == 0 and next_y < max_h:
                    y1 = i +1

            if prev_y != None:
                if y2 ==0 and prev_y <max_h:
                    y2 = i + 1

    for i, x in enumerate(v):
        if i > 0:
            prev_x = v[i - 1]
        if i + 1 < len(v):
            next_x = v[i + 1]

        if x == max_v:
            if next_x != None :
                if x1 == 0 and next_x < max_v:
                    x1 = i + 1
                    # x2 = i + 1
            if prev_x != None:
                if x2 == 0 and prev_x < max_v:
                    x2 = i +1
        if i == len(v)-1:
            if x2 == 0:
                x2 = image.shape[1]
            if y2 == 0:
                y2 = image.shape[0]

    crop_img = image[y1:y2, x1:x2 ]
    # binary_crop_img = image_to_binary(crop_img)
    # v = vertical_projection(binary_crop_img)
    # h = vertical_projection(binary_crop_img)
    # show(crop_img)
    return crop_img

    print(f'x1:{x1}, x2:{x2}, y1:{y1}, y2:{y2}')
    print("max_v: ", max_v)
    print("max_h: ", max_h)
    # print(h)
    # print(v)

def check_type_box(image):
    binary = image_to_binary(image)
    binary = image_to_binary(binary)
    # show(binary,"binary")
    blocks_projection = horizontal_projection(binary)
    blocks = split_blocks(blocks_projection)
    count =  np.count_nonzero(blocks_projection == 0)
    # print("blocks: ", blocks)
    # print("count: ", count)
    if count > 2 :
        print('line table')
        count_column = 0
        positions_column = []
        columns = []
        for index_block , block in enumerate(blocks):
            if index_block == 0:
                continue
            else:
                crop_block = image[block["start"] : block["start"] + (block["end"] - block["start"]), 0: image.shape[1]]
                clean_crop_block = clean_image(crop_block)
                # check_and_clean_border = check_border(clean_crop_block)
                # print(check_and_clean_border)
                max = clean_crop_block.shape[0] * 255
                # show(clean_crop_block)

                crop_binary = image_to_binary(clean_crop_block)
                column_projection = vertical_projection(crop_binary)
                # find columns
                start_index = None
                end_index = None

                splits = []
                for column_index, column in enumerate(column_projection):

                    prev_column = None
                    next_column = None
                    if column_index >0:
                        prev_column = column_projection[column_index - 1]
                    if column_index +1 < len(column_projection):
                        next_column = column_projection[column_index + 1]
                    if column == max:
                        if column_index == 0:
                            start_index = 0
                        elif column_index + 1 == len(column_projection):

                            if prev_column != None:
                                if prev_column < max:
                                    start_index = column_index -1
                            splits.append({
                                "start_x": start_index,
                                "end_x": column_index,
                                "start_y": block["start"],
                                "end_y": block["end"]
                            })
                        else:
                            if next_column !=None:
                                if next_column < max:
                                    splits.append({
                                        "start_x":start_index,
                                        "end_x": column_index,
                                        "start_y":block["start"],
                                        "end_y":block["end"]
                                    })
                                    start_index = None
                            if prev_column != None:
                                if prev_column < max:
                                    start_index = column_index -1
                columns.append(splits)

        # check if table?
        print('----->',columns)
        table_arr= []
        blocks_arr=[]

        for sub_array in columns:
            if len(sub_array) >2:
                print('column inside')
                table_arr.append(sub_array)
            else:
                print('block')
                blocks_arr.append(sub_array)


        # all_lengths_greater_than_2 = all(len(sub_array) > 2 for sub_array in table_arr)


        all_same_length = True
        if len(table_arr) >0:
            for i in range(len(table_arr[0])):
                lengths = [len(sub_array[i]) for sub_array in table_arr]
                if not all(length == lengths[0] for length in lengths):
                    all_same_length = False
                    break
        # print('table',all_same_length)
        if all_same_length:
            # print('table: ', table_arr)
            for index_column_table,column_table in enumerate(table_arr):
                for index_cell, cell in enumerate(column_table):
                    if index_cell+1 <len(column_table):

                        crop_cell = image[cell["start_y"]: cell["start_y"] +(cell["end_y"] - cell["start_y"]),cell["end_x"] :cell["end_x"] + (column_table[index_cell +1]["end_x"] - cell["end_x"])  ]
                        # handle cell
                        clean_crop_cell_border = clean_crop_border(crop_cell)

                        show(clean_crop_cell_border)
        else:
            # print("blocks table", table_arr)
            # show(crop_block)
            pass
        if len(blocks_arr) >0:
            # print("block arr" , blocks_arr)
            pass


        # now after we cut the rows we check from the second row (after titles) check if there is a vertical lines
            # IF ITS DOEST ITS BLOCKS TABLE
                # IF ITS THE SAME POSITION EXACTLY FROM ALL THE ROWS ITS MEAN ITS TABLE
                # ELSE ITS MEAN ITS A BLOCK TABLE

def clean_outside_box(image):
    binary = image_to_binary(image)
    # show(binary,'outside')

    vertical_blocks = vertical_projection(binary)
    horizontal_blocks = horizontal_projection(binary)
    height, width = image.shape
    x1 ,y1, x2 ,y2  = 0,0,0,height
    max_width = height * 255
    max_height = width * 255
    for index_block, block in enumerate(vertical_blocks):
        # print('------------- vertical -------------')
        # print("x1=",    x1, "x2=",  x2 ,"y1=",  y1, "y2=",  y2)
        prev_block ,next_block = None, None
        if index_block >0:
            prev_block = vertical_blocks[index_block -1]
        if index_block +1 < len(vertical_blocks):
            next_block  = vertical_blocks[index_block + 1]
        if block == 0:
            if next_block is not None:
                if next_block > 0:
                    if x1 == 0 and vertical_blocks[0] != max_width:
                        x1 = index_block + 1
            if prev_block is not None:
                if prev_block > 0 :
                    x2 = index_block
        if index_block +1 == len(vertical_blocks):
            if x2 == 0:
                x2 = index_block +1
            if block ==max_width:
                x2 = index_block +1

    for index_block, block in enumerate(horizontal_blocks):
        prev_block, next_block = None, None
        if index_block > 0:
            prev_block = horizontal_blocks[index_block - 1]
        if index_block + 1 < len(horizontal_blocks):
            next_block = horizontal_blocks[index_block + 1]
        if block == 0:
            if next_block is not None:
                if next_block:
                    if next_block > 0 and horizontal_blocks[0] != max_height:
                        if y1 == 0:
                            y1 = index_block + 1
                if prev_block:
                    if prev_block > 0:
                        y2 = index_block
        if index_block + 1 == len(horizontal_blocks):
            if y2 == 0:
                y2 = index_block
        crop = image[y1:y2, x1: x2]
    return  crop
    # blocks = split_blocks(projection)
    # print(blocks)

def clean_inside_box(image):
    binary = image_to_binary(image)
    # show(binary, 'inside')
    vertical_blocks = vertical_projection(binary)
    horizontal_blocks = horizontal_projection(binary)
    height, width = image.shape
    x1, y1, x2, y2 = 0, 0, 0, image.shape[0]
    max_height = height *255
    max_width = width *255
    for index_block, block in enumerate(vertical_blocks):
        prev_block, next_block = None, None
        if index_block > 0:
            prev_block = vertical_blocks[index_block - 1]
        if index_block + 1 < len(vertical_blocks):
            next_block = vertical_blocks[index_block + 1]
        if block == max_height:
            if next_block is not None:
                if next_block <max_height:
                    if x1 == 0:
                        x1 = index_block + 1
            if prev_block is not None:
                if prev_block < max_height:
                    x2 = index_block

        if index_block + 1 == len(vertical_blocks):
            if x2 == 0:
                x2 = index_block +1
            # if block == max_height:
            #     x2 = index_block +1

    for index_block, block in enumerate(horizontal_blocks):
        prev_block, next_block = None, None
        if index_block > 0:
            prev_block = horizontal_blocks[index_block - 1]
        if index_block + 1 < len(horizontal_blocks):
            next_block = horizontal_blocks[index_block + 1]
        if block == max_width:
            if next_block is not None:
                if next_block < max_width:
                    if y1 == 0:
                        y1 = index_block + 1
            if prev_block is not None:
                if prev_block < max_width:
                    y2 = index_block
        if index_block + 1 == len(horizontal_blocks):
            if y2 == 0:
                y2 = index_block +1
            if block == max_width:
                y2 = index_block +1
    crop = image[y1:y2, x1: x2]
    print("x1:",x1,"x2:",x2,"y1:",y1,"y2:",y2)
    # show(image_to_binary(crop))
    return crop

def cut_buy_stroke_horizontal(image):
    binary = image_to_binary(image)
    show(binary)
    horizontal_blocks = horizontal_projection(binary)
    print(horizontal_blocks)
    x1, y1 , x2, y2 = 0, 0, 0, 0
    max_width = image.shape[1] * 255
    print(max_width)

    pattern = np.array([0,max_width])

    def find_pattern_numpy(array, pattern):
        array = np.array(array)
        pattern = np.array(pattern)
        pattern_length = len(pattern)

        # Create a view of the array with a sliding window of the pattern's length
        sliding_windows = np.lib.stride_tricks.sliding_window_view(array, pattern_length)

        # Compare each window with the pattern
        matches = np.all(sliding_windows == pattern, axis=1)

        # Get the indices where matches are True
        indices = np.where(matches)[0]

        return indices

    arr = find_pattern_numpy(horizontal_blocks, pattern)
    print('===>',arr)
    for index_block , block in enumerate(horizontal_blocks):
        next_block ,prev_block = None, None
        if index_block >0:
            prev_block = horizontal_blocks[index_block -1]
        if index_block +1 < len(horizontal_blocks):
            next_block = horizontal_blocks[index_block + 1]

        if block == 0:
            if next_block is not None:
                if next_block > 0:
                    x2 = index_block + 1
                if prev_block > 0:
                    x1 = index_block + 1

def check_type(image):
    binary = image_to_binary(image)
    v = vertical_projection(binary)
    h = horizontal_projection(binary)
    height, width = binary.shape
    max_width = width * 255
    max_height = height * 255

    if (h[0], h[-1], v[0] , v[-1]) == (max_width, max_width,max_height,max_height):
        return True
    else:
        return False

def check_box_type(image): # 'box' | 'table-regular' | 'table-multi' | '' | 'box-input' | ''
    binary = image_to_binary(image)
    erode_x =erode_image(binary, y=50)
    # show(erode_x, "check type")
    ## find columns
    height, width = binary.shape
    probeility_of_table = 0
    max_width = width * 255
    max_height = height *255
    start_index = 0
    vertical = vertical_projection(erode_x)
    columns = []
    prev_v, next_v = None, None

    for index_v, v in enumerate(vertical):
        if index_v > 0:
            prev_v = vertical[index_v - 1]
        if index_v +1 < len(vertical):
            next_v = vertical[index_v +1 ]
        if v == max_height:
            if index_v + 1 == len(vertical):
                columns.append({
                    "start": index_v,
                    "end": index_v +1
                })
            else:
                if next_v and prev_v:
                    if next_v <max_height and prev_v <max_height:
                        columns.append({
                            "start": index_v,
                            "end": index_v + 1
                        })
                    else:
                        if next_v:
                            if next_v < max_height:
                                # end of the white
                                columns.append({
                                    "start": start_index,
                                    "end": index_v + 1
                                })
                        if prev_v:
                            if prev_v < max_height:
                                # start
                                start_index = index_v
                else:
                    if next_v:
                        if next_v <max_height:
                            # end of the white
                            columns.append({
                                "start":start_index,
                                "end":index_v +1
                            })
                    if prev_v:
                        if prev_v <max_height:
                            # start
                            start_index = index_v


    print("columns = ", len(columns))
    if len(columns) > 2:
        probeility_of_table +=1
    ## check the rows now

    erode_y = erode_image(binary, y=50)
    # show(erode_x, 'x')
    # show(erode_y, 'y')

im = read_image('page_24.png')
show(im)
outside = clean_outside_box(im)

show(outside, '----')
# check if there is a box
box = check_type(outside) # return True or False
print(box)

if box:
    # its a box
    box_type = check_box_type(outside) # 'box' | 'table-regular' | 'table-multi' | '' | 'box-input' | ''


# inside = clean_inside_box(outside)



# show(image_to_binary(outside))
# v_out = vertical_projection(outside)
# h_out = horizontal_projection(outside)
# inside_height, inside_width = outside.shape
# max_inside_height = inside_height *255
# max_inside_width = inside_width *255

inside_binary = image_to_binary(outside)
show(inside_binary)
erode_img = erode_image(inside_binary, x=80)
show(erode_img)
split_blocks_inside_projections = horizontal_projection(inside_binary)
split_blocks_inside = split_blocks(split_blocks_inside_projections)

for index_inside_block , inside_block in enumerate(split_blocks_inside):
    crop = outside[inside_block["start"] : inside_block["start"] + (inside_block["end"] - inside_block["start"]),0:outside.shape[1]]
    binary_crop = image_to_binary(crop)

    # binary_crop_block = image_to_binary(crop_block)
    erode_img = erode_image(binary_crop, x=80)

    show(erode_img)

    blocks_projection = horizontal_projection(erode_img)

    start, end = 0,None
    height, width = binary_crop.shape
    max_width = width * 255
    bbox_arr  = []
    start_black, start_white = 0 ,0
    for index_block, block in enumerate(blocks_projection):
        next_block, prev_block = None, None

        if index_block > 0 :
            prev_block = blocks_projection[index_block-1]
        if index_block +1 < len(blocks_projection):
            next_block = blocks_projection[index_block+1]

        if index_block + 1 == len(blocks_projection):
            if len(bbox_arr) >0:
                if bbox_arr[-1]["type_background"] == 'white':
                    # end black
                    bbox_arr.append({
                        "type_background": "black",
                        "start": start_black,
                        "end": index_block
                    })
                else:
                    # end white
                    bbox_arr.append({
                        "type_background": "white",
                        "start": start_white,
                        "end": index_block + 1
                    })
        elif block == max_width:
            if next_block is not None and prev_block is not None and (next_block == 0 and prev_block == 0):
                bbox_arr.append({
                    "type_background": "black",
                    "start": start_black,
                    "end": index_block
                })
                start_white = index_block
                bbox_arr.append({
                    "type_background": "white",
                    "start": start_white,
                    "end": index_block + 1
                })
                start_black = index_block + 1
            else:
                if next_block is not None:
                    if next_block == 0:
                        # is  a cut
                        bbox_arr.append({
                            "type_background": "white",
                            "start": start_white,
                            "end": index_block + 1
                        })
                        start_black = index_block + 1
                if prev_block is not None:
                    if prev_block == 0:
                        bbox_arr.append({
                            "type_background": "black",
                            "start": start_black,
                            "end": index_block
                        })
                        start_white = index_block

            # elif
            # if  prev_block is not None:
            #     if prev_block == max_width:
            #         bbox_arr.append({
            #             "start": start,
            #             "end": index_block
            #         })
            #     start = index_block

        # if index_block == 0:
        #     continue
        # else:
        # clean_crop_block = clean_image(crop_block)

    print("bbox_arr:   ",bbox_arr)
    for index_bbox, bbox in enumerate(bbox_arr):

        crop_block = crop[bbox["start"]: bbox["start"] + (bbox["end"] - bbox["start"]), 0: crop.shape[1]]
        if bbox["type_background"] == "white":
            crop_block  = image_to_binary(crop_block)
        show(image_to_binary(crop_block), "before ")
        clean_outside = clean_outside_box(crop_block)
        clean_inside = clean_inside_box(clean_outside)
        binary_clean = image_to_binary(clean_inside)
        v_clean = vertical_projection(binary_clean)
        h_clean = horizontal_projection(binary_clean)
        columns_arr = []
        # print(np.all(v_clean == 0))
        # print(np.all(h_clean == 0))
        # show(crop_block, "cleannn")
        show(image_to_binary(clean_inside), "cleannn")
        # if np.all(v_clean == 0) == False and np.all(h_clean == 0) == False:
        #     height, width = clean_inside.shape
        #     # check how much column and there is index
        #     max_height = height * 255
        #     for index_column, column in enumerate(v_clean):
        #         if index_column > 0:
        #             next_column = v_clean[index_column - 1]
        #         if index_column +1 < len(v_clean):
        #             prev_column = v_clean[index_column + 1]
        #
        #         if column == max_height:
        #             print(index_column)
        #             if next_column is not None and prev_column is not None:
        #                 if next_column < max_height and prev_column < max_height:
        #                     columns_arr.append({
        #                         "start": index_column,
        #                         "end": index_column +1
        #                     })
        #                 else:
        #                     if next_column is not None:
        #                         if next_column < max_height:
        #                             columns_arr.append({
        #                                 "start": start_index,
        #                                 "end": index_column + 1
        #                             })
        #                             end_index = index_column
        #                     if prev_column is not None:
        #                         if prev_column < max_height:
        #                             start_index = index_column
        #             else:
        #                 if next_column is not None:
        #                     if next_column < max_height:
        #                         columns_arr.append({
        #                             "start": start_index,
        #                             "end": index_column+1
        #                         })
        #                         end_index = index_column
        #                 if prev_column is not None:
        #                     if prev_column < max_height:
        #                         start_index = index_column
        #     print(columns_arr)
        #     print("dimension:", height, width)




        # print(11111)
        # clean the lines
        # binary_crop = image_to_binary(crop_block)
        # show(binary_crop)
        # binary_crop = image_to_binary(binary_crop)
        # outside_clean = clean_outside_box(crop_block)

        # inside_clean = clean_inside_box(crop_block)

        # show(inside_clean, "cleannn")
    # cut_buy_stroke_horizontal(crop)
    # show(crop)
#
# inside_height, inside_width = inside.shape
# max_inside_height = inside_height *255
# max_inside_width = inside_width *255
# v_out = vertical_projection(inside)
# h_out = horizontal_projection(inside)

print('')

