Model from last assignment ran for 40 epochs each in below scenarios

  - without L1/L2 norm
  - with L1 norm
  - with L2 norm
  - with L1 and L2 norm

#### Below are the history plots for model

!['Test accuracies'](images/history/test_acc.png "Test accuracies")
!['Test Losses'](images/history/test_losses.png "Test Losses")
!['Train accuracies'](images/history/train_acc.png "Train accuracies")
!['Train Losses'](images/history/train_losses.png "Train Losses")

#### misclassified images

!["Only L1 norm Regularization"](images/misclassified/l1.png "Only L1 norm Regularization")
---
!['Only L2 norm Regularization'](images/misclassified/l2.png "Only L2 norm Regularization")
---
!["Without L1 or L2 norm Regularization"](images/misclassified/none.png "Without L1 or L2 norm Regularization")
---
!['Both L1 and L2 norm Regularization'](images/misclassified/both.png "Both L1 and L2 norm Regularization")
---


#### Based on the plots
We can see that for this model, any Regularization not required.

L1 or L2 norm are added as a Regularization parameter, but the model seems to be fluctuating at same plateau with train accuracy not increasing. Implying that this model is only already fit without regularization and that **L1 or L2 norm doesn't seem to have any impact on this particular model.** We can see the impact when we have huge difference in training and test acc/loss.


#### Based on the misclassified images

 We can see that the model can benefit from below augmentations
- slight rotation
- slight shear
