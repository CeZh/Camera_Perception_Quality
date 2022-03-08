# Update configuration file


def configuration_update(configs, args):
    try:
        if configs['model_parameters']['depth'] != args.depth:
            original = configs['model_parameters']['depth']
            configs['model_parameters']['depth'] = args.depth
            print('update depth from %d to %d due to argument update' % (original, args.depth))
    except:
        print('attention model depth not updated')

    try:
        if configs['model_parameters']['heads'] != args.head:
            original = configs['model_parameters']['heads']
            configs['model_parameters']['heads'] = args.head
            print('update heads from %d to %d due to argument update' % (original, args.head))
    except:
        print('attention model head not updated!')

    try:
        if configs['superpixel_parameters']['segments'] != args.segments:
            original = configs['superpixel_parameters']['segments']
            configs['superpixel_parameters']['segments'] = args.segments
            print('update superpixel segments from %d to %d due to argument update' % (original, args.segments))
    except:
        print('no superpixel or superpixel arameters not updated!')

    try:
        if configs['superpixel_parameters']['att_dim']  != args.super_att_dim:
            original = configs['superpixel_parameters']['att_dim']
            configs['superpixel_parameters']['att_dim'] = args.super_att_dim
            print('update superpixel attention dimension from %d to %d due to argument update' % (original, args.super_att_dim))
    except:
        print('no superpixel or superpixel arameters not updated!')

    return configs