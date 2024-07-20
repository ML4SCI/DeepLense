Physics Informed Neural Network for Dark Matter Morphology

Image by Wikipedia and Deeplense simulations. Greetings, everyone! I am very excited to share the progress of my project, "Physics-Guided Machine Learning" for Google Summer of Code 2024. Over the past two months, I have been working diligently with ML4SCI on this project.
DeepLens: Physics-Guided Machine Learning
You can find my code, results, and all the plots on Github.
Special thanks to (my incredible mentors) Sergei Gleyzer, Pranath Reddy, Anna Parul, PG Guan, and Mike Toomey.
Introduction to Deeplense
DeepLense is a deep learning framework designed for searching particle dark matter through strong gravitational lensing. Now a question may arise what is strong gravitational lensing? and what does it have to do with Dark Matter?
Gravitational Lensing
Gravitational Lensing is a phenomenon which is caused when a massive celestial body (such as galaxy clusters etc.) causes sufficient curvature of spacetime causing the path of light around it to be visibly bent as if by a lens.
What is spacetime? According to Einstein's general theory of relativity, time and space are fused together into a single four-dimensional continuum(A continuum is a continuous and unbroken whole, where changes or divisions are gradual rather than abrupt) called spacetime.
and Extremely heavy objects in the universe, like galaxies, cause spacetime to bend significantly.
Bending of spacetime by heavy objects imaged by ESAWhen light from a distant source passes near a lensing galaxy (a massive galaxy that warps spacetime), the path of the light bends due to the curvature of spacetime, creating the gravitational lensing effect.
What is Strong Gravitational Lensing? It is a type of gravitational lensing where the gravitational field of a massive galaxy or any other celestial object is strong enough to produce multiple images of the source galaxy. This effect occurs because the strong gravitational field bends light from the distant object around the lensing galaxy, creating multiple distorted or duplicated images.
Strong Gravitational Lensing. SourceWhat does it have to do with Dark Matter?
Dark matter doesn't emit or absorb light, making it invisible to us. However, we can detect its presence through its gravitational effects. By observing more gravitational lensing than what we would expect from visible matter alone, we can infer the presence of additional unseen matter, which we attribute to dark matter.
Project 
Deep learning is one of the most crucial tools today, as many advancements are driven by neural networks. This project aims to develop a physics-informed neural network framework to infer the properties and distribution of dark matter in the lensing galaxy responsible for strong gravitational lensing.
Data
The data includes gravitational lensing images categorized into three classes representing the substructures of the lensing galaxy: Axion, Cold Dark Matter, and No Substructure.
As we can see, there is not much visible difference between the three classesSide Note: Physics Informed Transformers Models have previously been tested by the DeepLense team and have shown very good results.
My approach
My approach towards this task can be broken down into three major steps.
Physics Informed Preprocessing 
The goal of this project is to study dark matter through strong gravitational lensing. To achieve this, we can use a fundamental approach called Learning from Difference which is we analyze the abnormal distribution of the intensity of the map as  this distribution depends upon the Lensing galaxy and its substructures as they are causing the bending of light.
The intuition behind using (Imax/I)​​ is for contrast enhancement. The logarithm function is applied to compress and filter the dynamic range of intensities. Squaring is then used to ensure all intensities are positive, as negative intensities are not possible. Finally, normalization (which is optional) is achieved using the tanh function to map values to a desired range. If we perform the above transformation on each image then we get.
Source Image Reconstruction be Physics Informed Encoder
The Gravitational Lensing Equation maps the coordinates of the Source Galaxy and the Gravitational Lensing Images. They can be used for the purpose of lensing inversion in order to reconstruct the Source Image.
For the Singular Isothermal Sphere (SIS) model, the angular deviation is proportional to Einstein's angle and is directed along the angular coordinate theta.
Hence using this information we use the Encoder to Encode the magnitude of the angular deviation which is product of the Angular Diameter Distance( distance to the lens from the observer) and Einstein's Radius.
For the direction of this angular deviation vector, we have to first transform the Euclidian coordinates to angular coordinates then we have to perform the lensing inversion and then after we get the angular coordinates of the source image and then again we perform the transformation of the angular coordinates to the euclidean coordinates.
But there may occur two problems here 
The coordinates of the source galaxy calculated may fall outside the bounds of the grid. This is a common phenomenon, and since strong gravitational lensing also causes a magnification effect, we can address this by expanding the grid to accommodate the source galaxy.
Two or more different points on the observed lensing grid may fall on the same source coordinate. In strong gravitational lensing scenarios, it is possible for multiple different points on the observed lensing grid to correspond to the same source coordinate, leading to overlapping images of the same source(or multiple Lensing images). To address this issue in the project, I have implemented a method to average the intensities at overlapping coordinates.

The following code performs the lensing inversion while taking care of the above two cases.
class Physics():
    def __init__(self,mag = 1,min_angle = -3.232,max_angle = 3.232):
        self.source_mag = mag
        self.min_angle = min_angle
        self.max_angle = max_angle
         
    def image_to_source(self,image,centre = None,E_r = None,deflection=None, gradient=None):
        length, width = image.shape
        pixel_width = (self.max_angle-self.min_angle)/length
        if centre is None:
            centre = (length//2,width//2)
        centre_x = centre[0]
        centre_y = centre[1]

        range_indices_x = np.arange(-(centre_x-1),length-(centre_x-1))
        range_indices_y = np.arange(-(centre_y-1),width-(centre_y-1))

        x, y = np.meshgrid(range_indices_x, range_indices_y, indexing='ij')
        x,y = x*pixel_width,y*pixel_width
        
        r = np.sqrt(x**2 + y**2)
        mask = (r==0)
        r[mask] = 1

        if deflection is not None:
            if deflection.shape != (length, width):
                raise ValueError(f"The deflection should be of shape (2, {length}, {width}) but got {deflection.shape}")
            xdef = (deflection * x) / r
            ydef = (deflection * y) / r
        elif gradient is not None:
            if gradient.shape != image.shape:
                raise ValueError("The gradient and image should be of the same shape")
            #gradient = gradient*r
            xdef = np.gradient(gradient, axis=0)
            ydef = np.gradient(gradient, axis=1)
        elif E_r is not None:
            k = np.ones((length,width))*E_r
            xdef = (k * x) / r
            ydef = (k * y) / r
        else:
            raise ValueError("Both deflection and gradient cannot be None")

        bx = x - xdef
        by = y - ydef
        
        bx,by = bx/pixel_width,by/pixel_width
        
        bx = np.clip(bx + centre_x*self.source_mag, 0, length*self.source_mag - 1).astype(int)
        by = np.clip(by + centre_y*self.source_mag, 0, width*self.source_mag - 1).astype(int)
        
        sourceimage = np.zeros((length*self.source_mag,width*self.source_mag), dtype=float)
        counts = np.zeros_like(sourceimage, dtype=int)

        for i in range(length):
            for j in range(width):
                sourceimage[bx[i, j], by[i, j]] += image[i, j]
                counts[bx[i, j], by[i, j]] += 1

        average_mask = counts > 0
        sourceimage[average_mask] /= counts[average_mask]
        
        return sourceimage
Following is an example of the source image reconstruction taking 1 as Einstein's Angle. 
CNN as Decoder for Better Local Feature Extraction
After obtaining the source image and the distorted image (following preprocessing), my goal is to train the model to learn from the differences between these images. This is crucial for understanding dark matter, as the differences between lensing images are influenced by dark matter. The basic architecture of my model can be described in the following image.
GradCAM Visualization
Grad-CAM is a technique used to interpret a model by highlighting the regions of an image where the model is focusing to make its decision. Since we aim to understand the dark matter influencing the changes between classes, analyzing Grad-CAM images can help us identify the locations of dark matter. For example, the following Grad-CAM images illustrate this for three different classes.
We can see for the CDM class the GradCAM is more around the distorted area of the differential/preprocessed image even though the model is only fed with the image and not with the preprocessed image at all.Although deep learning models are often considered black boxes, making strong claims challenging, the Grad-CAM visualization shows that, despite being trained solely on lensing images, the model still focuses on the regions of distortion in the preprocessed image. This occurs even though the model was not directly provided with preprocessed images during training.
Results 
I have tested and compared three models with a similar number of parameters to ensure a fair comparison.
Resnet18 (11.68M)
Lensiformer (15.78M)
New Model (14.43M) (However even smaller version of it with around 9 Million parameters outperforms the above two)

The performance of the new model can be analyzed by the following Receiver Operating Characteristic curves.
Following are the plots for the comparison between the ResNet18 model, Lensiformer and the new model.
These plots are for the comparison between the model while training and testing We can clearly see from the above plots that the new model converged much faster compared to the Lensiformer and ResNet18. Additionally, the new model also achieved a lower Cross Entropy Loss.
F1 Score
The new model achieved an F1 score of 0.997, the highest among the three models. In comparison, the Lensiformer achieved an F1 score of 0.96, and ResNet18 achieved an F1 score of approximately 0.91 on the test dataset.
ROC-Curve
Furthermore, the new model achieves an ROC AUC score of 1, the highest among all models. The ROC AUC score is particularly important in this context as it demonstrates the model's capability to differentiate between the classes across various decision thresholds. This suggests that Lensiformer possesses a robust discriminatory power, which is imperative for applications in astrophysics where false positives can have significant implications.
Since there are three classes, the plot displays the micro and macro averages of the ROC AUC scores for these classes.
Conclusion
The new Architecture performs very well and so does the Lensiformer and the implementation of Grad-CAM shows very interesting results.
Future Goals
The future goals involve:
Grad-CAM and other explainable AI tools can be valuable for studying dark matter, as they help visualize and interpret model decisions. However, Grad-CAM is sensitive because it relies on gradients and weights that are randomly initialized, affecting its robustness.
Testing of these models on a harder dataset.
Implementing a physics-informed loss function helps guide the model towards correct convergence and better reconstruction of the source galaxy. This is achieved by incorporating physical principles and source profile information into the loss function.
