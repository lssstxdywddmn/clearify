# 图片清晰度初筛：可运行基线

当前仅处理清晰度。CPU 运行，不下载模型，不调用视觉语言模型。适合原图及 YOLO 给出的主体 ROI。

## 标签定义与验证状态

用户的三个标签是“清晰 / 细节不清 / 主体不可辨”。本初筛基线将它们临时映射为“高锐利度 / 中锐利度 / 严重模糊候选”。第三类不是主体语义不可辨的证明：目标太小、遮挡、低纹理、曝光等情况不能由一个锐利度分数区分。

30 和 100 是本方案的初始工程参数，**不是论文通用阈值，也没有通过用户业务图片标定**。输出始终标记 threshold_status=unvalidated_on_business_images。严重模糊候选或低对比度样本会 requires_review=true。不要将分数直接解释为分类概率。

## 安装与调用

若现有环境可以 import cv2、numpy，无需安装。否则：

```bash
python -m pip install numpy opencv-python-headless
```

```bash
python clarity.py /path/to/image.jpg
python clarity.py /path/to/image.jpg --roi 100 80 500 400
python clarity.py /path/to/image.jpg --low 30 --high 100
```

ROI 的格式是原图中的 x1 y1 x2 y2，右、下坐标不包含在区域内。模型检测框需先转换回原图坐标，取整并裁剪到图像范围。此实现使用 OpenCV 默认解码方向；带 EXIF 旋转的照片需确保 ROI 与解码后的坐标一致。读取后转换为 8 位灰度，不进行几何缩放、锐化或去噪。

## 明确算法

在原始尺寸的图像或主体区域上，以 0 至 255 的灰度值计算四邻域拉普拉斯：

L(x,y) = I(x-1,y) + I(x+1,y) + I(x,y-1) + I(x,y+1) - 4I(x,y)

去掉一圈边界，在剩余 N 个像素上计算：

S = sum((L - mean(L))^2) / N

S >= 100：清晰；30 <= S < 100：细节不清；S < 30：主体不可辨（仅为严重模糊代理）。代码对应 cv2.Laplacian(gray, cv2.CV_64F, ksize=1)[1:-1, 1:-1].var()。

低纹理区域或低曝光会给低分，噪声、过度锐化会给高分。该数学指标不会理解图像内容。输入分辨率、主体尺度和预处理发生变化时需重新验证阈值；不能把 0..1 浮点图像的阈值直接用于本实现。

## 怎样得到业务可信的三个标签

1. 明确判断整图还是特定主体；主体应先由检测器定位，或人工指定 ROI。没有检测到目标时不能直接标为不可辨。
2. 从真实图片中抽取三类人工样本，用本程序计算 S，检查是否能找到分隔三类的两个阈值。
3. 在同源内容、同一视频或连续帧不跨训练/验证集的前提下，选定阈值，再用独立验证集报告每类召回率、混淆矩阵及宏平均 F1。
4. 若三类分数明显交叠，说明这个分数不足以区分。此时用梯度、局部对比度、主体原始像素尺度等特征训练小分类器，或微调轻量 CNN。无法通过固定阈值解决的样本需保留复核。

可以用同一算法和数据集搜索两个阈值，但不能把搜阈值的数据同时作为最终验证数据。合成高斯模糊只适合验证程序逻辑，不足以标定真实语义可辨性。

## 对其他模型建议的核对

- BRISQUE 是统计特征加预训练 SVR；用户可以直接推理，但算法本身不是无需训练的分类器。
- BRISQUE、NIQE、PIQE 是综合质量指标，也会受到噪声和压缩等因素影响，不能直接当作清晰度三分类。
- MATLAB 提供这三个函数；标准 OpenCV contrib quality 模块提供 BRISQUE，但没有同等内置的 NIQE、PIQE，不能混为一谈。
- NIMA 输出质量/审美评分分布，不是用户三类标签的概率；需要核对权重训练数据和任务。
- PIQE 官方的 Excellent/Good/Fair/Poor/Bad 是综合质量分级，不是“主体不可辨”的标准。

## 测试

```bash
python -m unittest discover -s . -p test_clarity.py -v
```

已验证合成图的逐级模糊排序、低纹理复核标记、ROI 独立评估和无效参数检查。尚无用户真实样本，未验证业务三分类准确率。

## 资料

- 聚焦测量方法比较：https://www.sciencedirect.com/science/article/pii/S0031320312004736
- BRISQUE：https://www.mathworks.com/help/images/ref/brisque.html
- NIQE：https://www.mathworks.com/help/images/ref/niqe.html
- PIQE（含官方综合质量分级）：https://www.mathworks.com/help/images/ref/piqe.html
- OpenCV BRISQUE：https://github.com/opencv/opencv_contrib/blob/4.x/modules/quality/include/opencv2/quality/qualitybrisque.hpp
- NIMA：https://arxiv.org/abs/1709.05424
