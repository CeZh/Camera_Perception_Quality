import torch.nn as nn



class Perceptual_Quality_Estimation(nn.Module):
    def __init__(self, configs):
        super(Perceptual_Quality_Estimation, self).__init__()
        self.configs = configs

        # ViT Backbone w/ and w/o superpixel
        if configs['model_parameters']['backbone'] == 'ViT':
            from core.models import vit_backbone
            if configs['model_parameters']['use_superpixel']:
                self.backbone_module = vit_backbone.ViT(image_size=configs['model_parameters']['img_size'],
                                                        patch_size=configs['model_parameters']['patch_size'],
                                                        dim=configs['model_parameters']['patch_dim'],
                                                        depth=configs['model_parameters']['depth'],
                                                        heads=configs['model_parameters']['heads'],
                                                        mlp_dim=configs['model_parameters']['mlp_dim'],
                                                        emb_dropout=configs['model_parameters']['dropout_rate'],
                                                        super_pixel=configs['superpixel_parameters'],
                                                        channels=configs['model_parameters']['pixel_channel'])
            else:
                self.backbone_module = vit_backbone.ViT(image_size=configs['model_parameters']['img_size'],
                                                        patch_size=configs['model_parameters']['patch_size'],
                                                        dim=configs['model_parameters']['patch_dim'],
                                                        depth=configs['model_parameters']['depth'],
                                                        heads=configs['model_parameters']['heads'],
                                                        mlp_dim=configs['model_parameters']['mlp_dim'],
                                                        emb_dropout=configs['model_parameters']['dropout_rate'],
                                                        channels=configs['model_parameters']['pixel_channel'])


        # MLP Regression
        if configs['model_parameters']['regressor'] == 'MLP':
            from core.models import mlp_regressor
            self.regressor_module = mlp_regressor.MLP_Regressor(configs=configs,
                                                                output_dim=configs['model_parameters']['patch_dim'])


    def forward(self, x, **kwargs):
        if kwargs:
            super_x = kwargs['super_pixel']
            super_pos = kwargs['super_pos']
            back_out = self.backbone_module(x, super_x=super_x, super_pos = super_pos)
        else:
            back_out = self.backbone_module(x)

        output = {'regress': set(), 'class': set()}
        # Attention Network
        if self.configs['model_parameters']['regressor'] == 'Attention':
            module_output = self.attention_module(back_out)
            if self.configs['model_parameters']['mode'] == 'reg':
                regress_out = self.regressor(module_output)
                output['model_parameters']['regress'] = regress_out
            if self.configs['model_parameters']['mode'] == 'cls':
                classes_out = self.classifier(module_output)
                output['model_parameters']['class'] = classes_out
            if self.configs['model_parameters']['mode'] == 'cls+reg':
                regress_out = self.regressor(module_output)
                classes_out = self.classifier(module_output)
                output['model_parameters']['regress'] = regress_out
                output['model_parameters']['class'] = classes_out

        # MLP baseline
        elif self.configs['model_parameters']['regressor'] == 'MLP':
            regress_out = self.regressor_module(back_out)
            output['regress'] = regress_out

        return output

