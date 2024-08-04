import numpy as np
import cv2
import fitz
from numpy import binary_repr

root = {
    "paragraphs": [],
    "table":[]
}

from ocr import get_text_from_image
def read_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def image_to_binary(image,thresh = 250): # or 130
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

    return block_bbox

def split_blocks_table(blocks,max_width):

    bboxs = []

    start_index = 0
    end_index = 0

    for index_block, block in enumerate(blocks):
        prev_block, next_block = None, None
        if index_block + 1 < len(blocks):
            next_block = blocks[index_block + 1]
        if index_block > 0:
            prev_block = blocks[index_block - 1]
        if block == max_width:
            # CHECK NEXT
            if next_block is not None:
                if next_block == 0:
                    # end white start black
                    bboxs.append({
                        "type": "black",
                        "start": start_index,
                        "end": index_block + 1
                    })
                    start_index = index_block
                    pass
            if prev_block  is not None:
                if prev_block == 0:
                    # start white end black
                    bboxs.append({
                        "type":"white",
                        "start":start_index,
                        "end":index_block + 1
                    })
                    start_index = index_block
        else:
            if index_block + 1 == len(blocks):
                if bboxs[-1]["type"] == "black":
                    bboxs.append({
                        "type": "white",
                        "start": start_index,
                        "end": index_block + 1
                        })
                elif bboxs[-1]["type"] == "white":
                    bboxs.append({
                        "type": "black",
                        "start": start_index,
                        "end": index_block + 1
                    })
    print(bboxs)
    return bboxs

def split_pdf_to_images(path):
    pdf = fitz.open(path)
    for index_page in range(len(pdf)):
        page = pdf[index_page]
        pix = page.get_pixmap()
        pix.save(f'page_{index_page}.png')


def clean_image(image):

    binary = image_to_binary(image)
    blocks_projection = vertical_projection(binary)
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
    h = horizontal_projection(binary)
    v = vertical_projection(binary)
    prev_x, prev_y, next_x, next_y = None, None,None,None
    height, width = image.shape
    max_h = width * 255
    max_v = height * 255

    x1, y1,x2,y2= 0,0,0,0
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
    return crop_img

def check_type_box(image):
    binary = image_to_binary(image)
    binary = image_to_binary(binary)
    blocks_projection = horizontal_projection(binary)
    blocks = split_blocks(blocks_projection)
    count =  np.count_nonzero(blocks_projection == 0)
    if count > 2 :
        # print('line table')
        count_column = 0
        positions_column = []
        columns = []
        for index_block , block in enumerate(blocks):
            if index_block == 0:
                continue
            else:
                crop_block = image[block["start"] : block["start"] + (block["end"] - block["start"]), 0: image.shape[1]]
                clean_crop_block = clean_image(crop_block)
                max = clean_crop_block.shape[0] * 255
                crop_binary = image_to_binary(clean_crop_block)
                column_projection = vertical_projection(crop_binary)
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
        # print('----->',columns)
        table_arr= []
        blocks_arr=[]

        for sub_array in columns:
            if len(sub_array) >2:
                # print('column inside')
                table_arr.append(sub_array)
            else:
                # print('block')
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

                        # show(clean_crop_cell_border)
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
    # show(image, "tt")
    # print(height, width)
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
                    if x1 == 0 and vertical_blocks[0] != max_width and vertical_blocks[0] == 0:
                        x1 = index_block + 1
            if prev_block is not None:
                if prev_block > 0 :
                    x2 = index_block
        if index_block +1 == len(vertical_blocks):
            if x2 == 0:
                x2 = index_block + 1
            if block ==max_width or block > 0:
                x2 = index_block + 1

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
    # print('x1:', x1, 'y1:', y1, 'x2:', x2, 'y2:', y2)
    crop = image[y1:y2, x1: x2]
    # show(crop, "pp")
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
    # print("x1:",x1,"x2:",x2,"y1:",y1,"y2:",y2)
    # show(image_to_binary(crop))
    return crop

def cut_buy_stroke_horizontal(image):
    binary = image_to_binary(image)
    # show(binary)
    horizontal_blocks = horizontal_projection(binary)
    # print(horizontal_blocks)
    x1, y1 , x2, y2 = 0, 0, 0, 0
    max_width = image.shape[1] * 255
    # print(max_width)

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
    # print('===>',arr)
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


    # print("columns = ", len(columns))
    if len(columns) > 2:
        probeility_of_table +=1
    ## check the rows now

    erode_y = erode_image(binary, y=50)
    # show(erode_x, 'x')
    # show(erode_y, 'y')

def check_background(image):
    max_height = image.shape[1] * 255
    # print(max_width, max_height)
    erode_ = erode_image(image, x= 100)
    erode_binary = image_to_binary(erode_)
    h_clean = horizontal_projection(erode_binary)
    if np.all(h_clean == max_height):
        return True
    return False
def check_box(image):

    box = check_type(image)

    if box:
        # its a box
        box_type = check_box_type(image) # 'box' | 'table-regular' | 'table-multi' | '' | 'box-input' | ''
        # inside = clean_inside_box(image)
        # show(image)
        erode_ = erode_image(image, x= 100)
        binary_ = image_to_binary(image, thresh=100)
        erode_binary =  image_to_binary(erode_)
        h_clean = horizontal_projection(erode_binary)
        # print(h_clean)
        # show(erode_, "erode")

        max_width = image.shape[0] * 255
        max_height = image.shape[1] * 255
        # print(max_width, max_height)
        if np.all(h_clean == max_height):
            # print('background text is black')
            # binary_ = image_to_binary(binary_)
            # show(binary_)
            split_rows(binary_)
        else:
            print('box | input | table | table-box etc .. ')


    else:
        inside_binary = image_to_binary(image)
        split_rows(inside_binary)
        # show(inside_binary)
        #
        # split_blocks_projections = horizontal_projection(inside_binary)
        # splt_blocks = split_blocks(split_blocks_projections)
        # # print(splt_blocks)
        # for index_block, block in enumerate(splt_blocks):
        #     crop = image[block["start"]: block["start"] + (block["end"]- block["start"]), 0:image.shape[1]]
        #     clean_crop = clean_outside_box(crop)
        #     # show(clean_crop, "line")
        #
        #     check_box = check_type(clean_crop)
        #     if check_box:
        #         # print('box')
        #         check_box(clean_crop)
        #     else:
        #         # extract rows
        #         # print(get_text_from_image(clean_crop))
        #         # print('not table')
        #         pass
im = read_image('page_27.png')
def split_columns_table(columns, max_column):
    columns_arr = []
    for index_px, px in enumerate(columns):
        prev_px,next_px = None, None
        if index_px >0:
            prev_px =columns[index_px -1]
        if index_px +1 < len(columns):
            next_px = columns[index_px+1 ]

        if px == max_column:
            if prev_px is not None:
                if prev_px< max_column:
                    start_px = index_px
            if next_px is not None:
                if next_px < max_column:
                    end_px = index_px + 1
                    columns_arr.append({"start": start_px, "end": end_px})

    return  columns_arr

def handle_box(image):
    clean_inside = clean_inside_box(image)
    # show(clean_inside, 'inside')
    binary = image_to_binary(clean_inside)
    # show(binary)
    h = horizontal_projection(image_to_binary(binary))
    erode = erode_image(binary , x =100)
    # show(erode, "e")
    h_ = horizontal_projection(erode)
    height, width = erode.shape
    rows_ = split_blocks_table(h_,width *255)
    table = []

    for index_row, row in enumerate(rows_):
        crop_row = clean_inside[row["start"]:row["start"]+ (row["end"] - row["start"]),0:clean_inside.shape[1]]
        check_back = check_background(crop_row)
        all_divide = np.all(horizontal_projection(image_to_binary(image_to_binary(crop_row))) == 0)
        if not all_divide:
            if check_back:
                # print(True)
                # flip color
                crop_row = image_to_binary(crop_row, 100)
                # show(crop_row)
            binary = image_to_binary(crop_row)
            v = vertical_projection(binary)
            columns_spl = split_columns_table(v, binary.shape[0] * 255)
            for index_column, column_spl in enumerate(columns_spl):
                row = []
                if index_column == 0:
                    crop_start = crop_row[0:crop_row.shape[0], 0:column_spl["start"]]
                    row.append({
                        "type": "cell",
                        "text": get_text_from_image(crop_start)
                    })
                if index_column + 1 == len(columns_spl):
                    crop_end = crop_row[0:crop_row.shape[0],
                               column_spl["end"]:column_spl["end"] + (crop_row.shape[1] - column_spl["end"])]
                    row.append({
                        "type": "cell",
                        "text": get_text_from_image(crop_end)
                    })
                if index_column > 0 and index_column + 1 < len(columns_spl):
                    crop_middle = crop_row[0:crop_row.shape[0],
                                  columns_spl[index_column - 1]["end"]: columns_spl[index_column - 1]["end"] + (
                                              column_spl["start"] - columns_spl[index_column - 1]["end"])]
                    row.append({
                        "type": "cell",
                        "text": get_text_from_image(crop_middle)
                    })

                table.append(row)
    root["table"].append({
        "id": len(root["table"]),
        "data": table
    })


    # rows = split_blocks(h)
    #
    # for index_row, row in enumerate(rows):
    #     # print(row)
    #     crop_row = clean_inside[row["start"]:row["start"]+ (row["end"] - row["start"]),0:clean_inside.shape[1]]
    #     show(crop_row, "crop_row")
    #     print(check_background(crop_row))
    #
    #     check_back = check_background(crop_row)
    #
    #     if check_back:
    #         crop_row = image_to_binary(crop_row, 190)
    #         show(crop_row)
    #
    #     if index_row > -1:
    #         binary = image_to_binary(crop_row)
    #         show(binary)
    #         v = vertical_projection(binary)
    #         columns_spl = split_columns_table(v,binary.shape[0] *255)
    #         table = []
    #
    #         for index_column, column_spl in enumerate(columns_spl):
    #             row = []
    #             if index_column == 0:
    #                 crop_start = crop_row[0:crop_row.shape[0], 0:column_spl["start"]]
    #                 row.append({
    #                     "type":"cell",
    #                     "text":get_text_from_image(crop_start)
    #                 })
    #                 # print(get_text_from_image(crop_start))
    #                 # show(crop_start)
    #             if index_column + 1 == len(columns_spl):
    #                 crop_end = crop_row[0:crop_row.shape[0], column_spl["end"]:column_spl["end"]+(crop_row.shape[1] - column_spl["end"])]
    #                 # show(crop_end)
    #                 # print(get_text_from_image(crop_end))
    #                 row.append({
    #                     "type": "cell",
    #                     "text": get_text_from_image(crop_end)
    #                 })
    #             if index_column >0 and index_column + 1 < len(columns_spl):
    #                 crop_middle = crop_row[0:crop_row.shape[0], columns_spl[index_column - 1]["end"]: columns_spl[index_column - 1]["end"] + (column_spl["start"] - columns_spl[index_column - 1]["end"]) ]
    #                 show(crop_middle)
    #                 row.append({
    #                     "type":"cell",
    #                     "text":get_text_from_image(crop_middle)
    #                 })
    #             root["table"].append(row)
    #

def split_rows(im):

    outside = clean_outside_box(im)
    inside_binary = image_to_binary(outside)
    # show(inside_binary)
    split_blocks_inside_projections = horizontal_projection(inside_binary)
    split_blocks_inside = split_blocks(split_blocks_inside_projections)

    for index_inside_block, inside_block in enumerate(split_blocks_inside):

        crop = outside[inside_block["start"]: inside_block["start"] + (inside_block["end"] - inside_block["start"]),
               0:outside.shape[1]]
        clean_crop = clean_outside_box(crop)
        binary_crop = image_to_binary(clean_crop)
        # show(binary_crop)
        # print("")
        box = check_type(clean_crop)

        if box:
            # print('box')
            handle_box(clean_crop)
        else:
            erode_img = dilate_image(binary_crop, x=20)
            vertical_column_projection = vertical_projection(erode_img)
            split_blocks_columns = split_blocks(vertical_column_projection)
            len_columns = len(split_blocks_columns)
            # print("length columns: ", len_columns)
            if len_columns == 1:
                # print('one column optional row')
                # show(clean_crop)
                crop_column_ = clean_crop[0:clean_crop.shape[0], split_blocks_columns[0]["start"]: split_blocks_columns[0]["start"] + (
                            split_blocks_columns[0]["end"] - split_blocks_columns[0]["start"])]
                # show(crop_column_)
                # show(image_to_binary(crop_column_))
                # print("text:", get_text_from_image(crop_column_))
                root["paragraphs"].append({
                    "type": "line",
                    "text": get_text_from_image(crop_column_)
                })
            else:
                for index_column_block, inside_column in enumerate(split_blocks_columns):
                    crop_column = clean_crop[0:clean_crop.shape[0],inside_column["start"]: inside_column["start"] + (inside_column["end"] - inside_column["start"])]
                    binary_crop = image_to_binary(crop_column)

                    clean_crop_column = clean_outside_box(crop_column)
                    box = check_type(clean_crop_column)
                    if box:
                        # print('box')
                        handle_box(clean_crop_column)
                    else:
                        split_rows(crop_column)

split_rows(im)
print(root)

