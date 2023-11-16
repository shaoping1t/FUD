python tools/train.py -c ./configs/rec/rec_fud_tiny_ch.yml


python -m paddle.distributed.launch tools/train.py -c configs/rec/rec_fud_tiny_ch.yml


python tools/eval.py -c ./configs/rec/rec_fud_tiny_ch.yml



cd C:\Users\w\PycharmProjects\FUD

activate paddleocr


python tools/infer_rec.py -c configs/rec/rec_fud_tiny_ch.yml  -o  Global.infer_img=E:/infer/scene/  Global.use_gpu=True


