from torch.utils.data import Dataset


class SARDData(Dataset):

    def __init__(self, indices, outcomes, linear_predictions=None, stage='train',
                 distill=True):
        """

        Parameters
        ----------
        indices : dict          with train, val and test indices
        outcomes :              outcome labels
        linear_predictions :    predictions from previous model to distill
        stage : str             needs to be one of "train", "validation" or "test"
        distill :               if run for distillation or not, if distillation then get_item returns also predictions
                                of already fit model
        """

        self.distill = distill
        self.train_indices = indices['train']
        self.validation_indices = indices['val']
        self.test_indices = indices['test']

        self.outcomes = outcomes
        self.linear_predictions = linear_predictions
        if stage == 'train':
            self.indices = self.train_indices
        elif stage == 'validation':
            self.indices = self.validation_indices
        elif stage == 'test':
            self.indices = self.test_indices
        else:
            raise ValueError(f'stage is not correct, needs to be one of: "train", "validation" or "test" but {stage}'
                             f' was provided')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self.distill:
            return self.indices[item], (self.outcomes[self.indices[item]], self.linear_predictions[self.indices[item]])
        else:
            return self.indices[item], self.outcomes[self.indices[item]]



