class DataLoader:
    def __init__(self, image_data_path, train_data_txt, test_data_txt, img_size, normalize):
        self.training_img_path = image_data_path
        self.data_csv = train_data_txt
        self.img_size = img_size
        self.test_data_txt = test_data_txt
        self.normalize = normalize

    def load_image(self, filename):
        img = Image.open(filename)
        width, height = self.img_size
        img = img.resize((height, width))
        img = ImageOps.equalize(img, mask=None)
        img_arr = np.asarray(img, dtype=np.float32)
        
        if self.normalize:
            img_arr = img_arr / 255
            
        return img_arr

    def load_test_datas(self):
        print("Loading test datas ...")
        X_image = self.load_images(self.training_img_path, self.test_data_txt)
        print("Done !")
        return X_image

    def load_train_datas(self):
        print("Loading train datas ...")
        X_image = self.load_images(self.training_img_path, self.data_csv)
        y = self.load_labels(self.data_csv)
        print("Done !")

        return X_image, y

    def load_images(self, training_img_path, data_csv):
        X_data = pd.read_csv(
            data_csv, sep=" ", header=None, names=["filename", "label"]
        )
        filename = X_data["filename"].values
        h, w = self.img_size[0], self.img_size[1]
        dataimage = np.zeros((len(X_data), h, w, 3), dtype=np.float32)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = [
                executor.submit(self.load_image, training_img_path + str(filename[i]))
                for i in range(len(X_data))
            ]

            for i, image in tqdm(enumerate(executor.map(lambda x: x.result(), tasks))):
                dataimage[i] = image

        dataimage = dataimage / 255  # normalize images
        dataimage = np.array(dataimage)

        return dataimage

    def load_labels(self, data_csv):  # pour train
        y = pd.read_csv(data_csv, sep=" ", header=None, names=["filename", "label"])
        y.drop(y.columns[[0]], axis=1, inplace=True)
        return np.array(y)

    def data_aug(self, X, y, n):
        augmented_data = []
        target = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(X.shape[0]):
                futures.append(
                    executor.submit(
                        self._augment_data, X[i], y[i], n, augmented_data, target
                    )
                )
            concurrent.futures.wait(futures)
        X = np.array(augmented_data)
        y = np.array(target)
        return X, y

    def _augment_data(self, x, y, n, augmented_data, target):
        augmented_data.append(x)
        target.append(y)
        augmented_data.append(np.fliplr(x))
        augmented_data.append(np.flipud(x))
        for j in range(n):
            target.append(y)


if __name__ == "__main__":
    
    width, height = (60, 60)
    train_data_txt_path = "../Documents/img_esiea/train_2022_DL.txt"
    images_path = "../Documents/img_esiea/"
    test_data_txt_path = "../Documents/img_esiea/valid_2022_DL.txt"
    
    data = DataLoader(
        image_data_path=images_path,
        train_data_txt=train_data_txt_path,
        test_data_txt=test_data_txt_path,
        img_size=(width, height),
        normalize = True,
    )

    X, y = data.load_train_datas()
    X_test = data.load_test_datas()
    np.save(f"../Documents/X_{str(width)}.npy", X)
    np.save(f"../Documents/y_{str(width)}.npy", y)
    np.save(f"../Documents/X_test_{str(width)}.npy", X_test)
    
    import sys

    for var_name in dir():
        if not var_name.startswith("_"):
            del globals()[var_name]
    import gc
    gc.collect()
    
    X, y = np.load(f"../../Documents/X_{str(width)}.npy"), np.load(
    f"../../Documents/y_{str(width)}.npy"
    )
    X_test = np.load(f"../../Documents/X_test_{str(width)}.npy")
