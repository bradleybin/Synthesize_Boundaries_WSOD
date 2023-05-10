import torch
import torch.nn.functional as F


class LocalSaliencyCoherence_attention(torch.nn.Module):
    """
    This loss function based on the following paper.
    Please consider using the following bibtex for citation:
    @article{obukhov2019gated,
        author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
        title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
        journal={CoRR},
        volume={abs/1906.04651},
        year={2019},
        url={http://arxiv.org/abs/1906.04651},
    }
    """

    #loss2_lsc = \
    #loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']

    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input, attention_map,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):

        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape

        device = y_hat_softmax.device

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        xy_features, rgb_features = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers   
        )
        xy_features = LocalSaliencyCoherence_attention._create_diff_from_features(xy_features, kernels_radius)
        rgb_features = LocalSaliencyCoherence_attention._create_diff_from_features(rgb_features, kernels_radius) * (LocalSaliencyCoherence_attention._create_diff_from_features(attention_map, kernels_radius))

        kernels = torch.cat([rgb_features, xy_features], dim=1)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()


        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        y_hat_unfolded = torch.abs(y_hat_unfolded[:, :, kernels_radius, kernels_radius, :, :].view(N, C, 1, 1, height_pred, width_pred) - y_hat_unfolded) 

        loss = torch.mean((kernels * y_hat_unfolded).view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred).sum(dim=2, keepdim=True)) 


        out = {
            'loss': loss.mean(),
        }

        if out_kernels_vis:
            out['kernels_vis'] = self._visualize_kernels(
                kernels, kernels_radius, height_input, width_input, height_pred, width_pred
            )

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        #kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            rgb_features = []
            xy_features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':   
                    feature = LocalSaliencyCoherence_attention._get_mesh(N, height_pred, width_pred, device)
                    feature /= sigma
                    xy_features.append(feature)
                else:          
                    assert modality in sample, \
                        f'Modality {modality} is listed in {i}-th kernel descriptor, but not present in the sample'
                    feature = sample[modality]

                    feature /= sigma
                    rgb_features.append(feature)
               
            rgb_features = torch.cat(rgb_features, dim=1)
            xy_features = torch.cat(xy_features, dim=1)

            #kernel = weight * LocalSaliencyCoherence._create_kernels_from_features(features, kernels_radius)
            #kernels = kernel if kernels is None else kernel + kernels

        return xy_features, rgb_features

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = LocalSaliencyCoherence_attention._unfold(features, radius)     
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W) 
        #(N, C, diameter, diameter, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        # kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _create_diff_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = LocalSaliencyCoherence_attention._unfold(features, radius)   
        diff_map = torch.abs(kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)) 

        return diff_map

    #create pixels around edges

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),       
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)   
                #
                #unfold(input, kernel_size, dilation=1, padding=0, stride=1)  

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred-vis.shape[3], 0, height_pred-vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis
