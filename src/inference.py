import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    model = models.resnet18().cuda()
    inputs = torch.randn(5, 3, 224, 224).cuda()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    prof.export_chrome_trace('trace.json')


if __name__ == '__main__':
    main()
