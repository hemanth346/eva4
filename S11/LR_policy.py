class LrPolicy(object):
    def __init__(self, min_lr=0.0001, max_lr=5):
        self.min_lr = min_lr
        self.max_lr = max_lr

    def traingular(self, iterations, step_size, plot=True):
        min_lr, max_lr = self.min_lr, self.max_lr
        lin_space = (max_lr - min_lr)/(step_size-1)
        print('LR Change per iteration : (+/-)',lin_space)
        lr_array = []
        lr = min_lr
        for i in range(1, iterations+1):
            # downward slope on even step_size
            if not i%step_size: # i%step_size == 0
                lin_space = -lin_space
            lr_array.append(lr)
            lr += lin_space        
        if plot:
            self.plot_lr(lr_array)
        else:
            return lr_array

    @staticmethod
    def plot_lr(lr_array):
        plt.plot(lr_array)
        plt.xlabel('Iterations')
        plt.ylabel('LR')
        plt.title('LR Policy')
        plt.show()
