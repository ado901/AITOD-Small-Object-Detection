from funcs import MODELS
# in run.py based on the model chosen, the different parameters are loaded
epochs=100
batchsize=8
step_lr_params={'step_size':3,'gamma':0.1}
reduceLRonplateu_params='default'
params_retinanet={'optimizer':{'name':'SGD','params':{'lr':0.01,'weight_decay':0.0001}},
                               'scheduler':{'name':'reduceLRonplateu','params':reduceLRonplateu_params},
                               'model':{ "name": MODELS.RETINANET_RESNET.__repr__().split(' ')[1], 'params':{'pretrained':True}},
                               'image_repr':'default',
                               'batch_size':batchsize,
                               'epochs':epochs}
params_faster_rcnn_resnet={'optimizer':{'name':'SGD','params':{'lr':0.01,'weight_decay':0.0001}},
                           'scheduler':{'name':'reduceLRonplateu','params':reduceLRonplateu_params},
                           'model':{ "name": MODELS.FASTER_RCNN_RESNET.__repr__().split(' ')[1], 'params':{'pretrained':True}},
                            'image_repr':'default',
                            'batch_size':batchsize,
                            'epochs':epochs}
params_custom_rpn_faster_rcnn_resnet={'optimizer':{'name':'SGD','params':{'lr':0.01,'weight_decay':0.0001}},
                            'scheduler':{'name':'reduceLRonplateu','params':reduceLRonplateu_params},
                            'model':{ "name": MODELS.FASTER_RCNN_RESNET_RPN.__repr__().split(' ')[1], 'params':{'rpn':{ "sizes":((4,), (8,), (16,), (32,), (64,),), "aspect_ratios":((0.5, 1.0, 2.0),)},"pretrained":False}},
                             'image_repr':'default',
                             'batch_size':batchsize,
                             'epochs':epochs}

        