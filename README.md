# Breast Cancer Image Classifier
- M. Beekenkamp.

## Introduction
Breast cancer is the most common cancer worldwide, accounting for $12.5\%$ of all newly diagnosed cancers globally in $2020$ and $30\%$ in America in 2023, according to Melina Arnold et al. and the American Cancer Society respectively.
Mammography is a type of medical imaging that uses a low-dose X-ray system to visualise the internal structure of breast tissue and is the most effective way to detect breast cancer in its early stages. 
The early detection of breast cancer allows for easier treatment options, providing better healthcare outcomes by improving survival rates and decreasing the morbidity associated with the disease. 
Mammography screening has its limitations. Radiologists may miss subtle signs of breast cancer in mammograms, leading to false-negative results and delayed diagnosis. 
Computer-aided detection (CAD), which employs image processing techniques and pattern recognition theory to detect features like micro-calcifications, areas of increased density, and areas of asymmetry, has been introduced to provide an objective view to radiologists by marking possible areas of concern. 
There is a clear potential for the large-scale clinical application of CAD models; however, accurate detection of breast cancer has remained challenging. 
After the use of CAD models was approved in 2002 by the Centers for Medicare \& Medicaid Services, studies have shown that traditional CAD models have high false positive rates. 
False-positive results lead to anxiety for patients, and unnecessary biopsies, which in America is accompanied by significant financial burdens. <br>

This is why I applied my knowledge in machine learning to this problem, in the hope that with new technology better patient outcomes could be achieved.

## Results
Using the [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) this code successfully assigns labels representing the diagnosis (benign or malignant) 96.49% of the time.



## Socially Responsible Computing Implications
The development and implementation of deep learning frameworks for assisting radiologists in breast cancer detection raise important ethical considerations that must be carefully addressed. 
Firstly, there's a concern regarding patient privacy and data security, as the project involves handling large datasets of sensitive medical information. 
Ensuring robust data anonymization techniques and strict adherence to data protection regulations is essential to safeguard patient confidentiality. 
For this project, as previously mentioned, the data was obtained from the Breast Cancer Wisconsin (Diagnostic) dataset. <br>

Secondly, there's a risk of algorithmic bias, where the model may inadvertently learn and perpetuate existing biases present in the training data. 
This bias could disproportionately affect certain demographic groups, leading to disparities in healthcare outcomes. 
It's crucial to mitigate bias through careful dataset curation, algorithmic transparency, and ongoing monitoring and evaluation of model performance across diverse populations. <br>

Moreover, the introduction of machine learning algorithms into medical decision-making processes raises questions about accountability and liability in the event of errors or adverse outcomes. 
While algorithms like these can provide valuable insights and support to healthcare professionals, they should be viewed as decision support tools rather than replacements for clinical judgment. 
Establishing clear guidelines for the responsible use of these tools, along with appropriate training for healthcare providers, is vital to ensure safe and effective integration into clinical practice.
Were this a bigger project it would be essential to involve healthcare professionals in the development process and provide adequate training and support to ensure socially responsible AI systems.

## Motivating Literature and Sources

Melina Arnold et al. [“Current and future burden of breast cancer: Global statistics for 2020 and 2040”](https://www.sciencedirect.com/science/article/pii/S0960977622001448). In: The Breast 66 (2022), pp. 15–23. issn: 0960-9776. doi: https://doi.org/10.1016/j.breast.2022.08.010
American Cancer Society medical and editorial content team. [Breast cancer statistics: How common is breast cancer?](https://www.cancer.org/cancer/breast-cancer/about/how-common-is-breast-cancer.html) Jan. 2023.
