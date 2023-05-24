import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms


class CarColorResnet18():
    def __init__(self):
        path_prototxt = "models/vehicle_information/Secondary_CarColor/resnet18.prototxt"
        path_caffeModel = "models/vehicle_information/Secondary_CarColor/resnet18.caffemodel"
        path_label = "models/vehicle_information/Secondary_CarColor/labels.txt"
        self.dnn_model = cv2.dnn.readNetFromCaffe(
            prototxt=path_prototxt,
            caffeModel=path_caffeModel
        )
        self.dnn_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.dnn_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.labels = np.loadtxt(path_label, str, delimiter=';')

    def classifier(self, image):
        preprocessed_image = cv2.dnn.blobFromImage(
            image, scalefactor=1.0, size=(224, 224), mean=(104.0, 117.0, 123.0),
            swapRB=False, crop=False)
        self.dnn_model.setInput(preprocessed_image)
        results = self.dnn_model.forward()
        color_name = self.labels[results.argmax()]
        return color_name


class CarMakeResnet18():
    def __init__(self):
        path_prototxt = "models/vehicle_information/Secondary_CarMake/resnet18.prototxt"
        path_caffeModel = "models/vehicle_information/Secondary_CarMake/resnet18.caffemodel"
        path_label = "models/vehicle_information/Secondary_CarMake/labels.txt"
        self.dnn_model = cv2.dnn.readNetFromCaffe(
            prototxt=path_prototxt,
            caffeModel=path_caffeModel
        )
        self.dnn_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.dnn_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.labels = np.loadtxt(path_label, str, delimiter=';')

    def classifier(self, image):
        preprocessed_image = cv2.dnn.blobFromImage(
            image, scalefactor=1.0, size=(224, 224), mean=(104.0, 117.0, 123.0),
            swapRB=False, crop=False)
        self.dnn_model.setInput(preprocessed_image)
        results = self.dnn_model.forward()
        brand_name = self.labels[results.argmax()]
        return brand_name


class CarColorResnet18():
    def __init__(self):
        self.model = torch.load("models/vehicle_information/Car_color_RegNet_Y_1_6GF_update_mydata.pt")
        self.model.eval()
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                #         transforms.TrivialAugmentWide(),
                #         transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                #         transforms.RandAugment(),
                #         transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.class_names = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple',
                            'red', 'silver', 'tan', 'white', 'yellow']

    def classifier(self, image):
        transform = self.data_transforms['test']
        PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
        test_image_tensor = transform(PIL_image)
        if torch.cuda.is_available():
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
        else:
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
        with torch.no_grad():
            self.model.eval()
            # Model outputs log probabilities
            out = self.model(test_image_tensor)
            ps = torch.exp(out)
            topk, topclass = ps.topk(1, dim=1)

        return self.class_names[topclass.cpu().numpy()[0][0]]


class CarBrand():
    def __init__(self):

        self.model = torch.load(r"C:\Users\maintn\PycharmProjects\pythonProject\Car_Mode_epoch_8.pt", map_location=torch.device('cpu'))
        self.model.eval()

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                #         transforms.TrivialAugmentWide(),
                #         transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                #         transforms.RandAugment(),
                #         transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                # transforms.Resize(256),
                transforms.Resize((224, 224)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        # self.class_names = ['ASTON DB11', 'ASTON Vantage', 'AUDI A3', 'AUDI A4', 'AUDI A5', 'AUDI A6', 'AUDI A7', 'AUDI A8', 'AUDI Q2', 'AUDI Q3', 'AUDI Q5', 'AUDI Q7', 'AUDI Q8', 'AUDI TT Coupé', 'BAIC Beijing X7', 'BENTLEY Bentayga', 'BENTLEY Continental', 'BENTLEY Flying Spur', 'BENTLEY Mulsannel', 'BMW BMW 1 Series 5 Door', 'BMW BMW 2 Series Gran Tourer', 'BMW BMW 3 Gran Turismo', 'BMW BMW 3 Series sedan', 'BMW BMW 4 Series Coupé', 'BMW BMW 5 Series Sedan', 'BMW BMW 7 Series Sedan', 'BMW X1', 'BMW X2', 'BMW X3', 'BMW X3 xDrive30e', 'BMW X4', 'BMW X4 M40i', 'BMW X5', 'BMW X6', 'BMW X7', 'BRILLIANCE V7', 'CHEVROLET Chevrolet Colorado', 'CHEVROLET Trailblazer', 'FORD EcoSport', 'FORD Everest', 'FORD Explorer', 'FORD Focus', 'FORD Ranger', 'FORD Transit', 'HONDA Accord', 'HONDA Brio', 'HONDA CR-V', 'HONDA City', 'HONDA Civic', 'HONDA HR-V', 'HONDA Jazz', 'HONGQI E-HS9', 'HONGQI H9', 'HYUNDAI Accent', 'HYUNDAI County', 'HYUNDAI Elantra', 'HYUNDAI Kona', 'HYUNDAI Santafe', 'HYUNDAI Solati', 'HYUNDAI Tucson', 'HYUNDAI Universe', 'HYUNDAI i10', 'ISUZU D-MAX', 'ISUZU MU-X B7', 'JAGUAR E-Pace', 'JAGUAR F-Pace', 'JAGUAR F-Type', 'JAGUAR I-Pace', 'JAGUAR SV', 'JAGUAR XE', 'JAGUAR XF', 'KIA Cerato', 'KIA Morning', 'KIA Optima', 'KIA Quoris', 'KIA Rondo', 'KIA Sedona', 'KIA Soluto', 'KIA Sorento', 'LAMBORGHINI Aventador LP 610-4', 'LAMBORGHINI Aventador LP 700-4', 'LAMBORGHINI Aventador S', 'LAMBORGHINI Huracan LP 580-2', 'LAMBORGHINI Huracan Performante', 'LAMBORGHINI Urus', 'LAND ROVER Defender', 'LAND ROVER Discovery', 'LAND ROVER Discovery Sport', 'LAND ROVER Evoque', 'LAND ROVER New Evoque', 'LAND ROVER Range Rover', 'LAND ROVER Sport', 'LAND ROVER Velar', 'LEXUS ES', 'LEXUS LC', 'LEXUS LS', 'LEXUS LS 500', 'LEXUS LS 500h', 'LEXUS LX', 'LEXUS NX', 'LEXUS NX 300', 'LEXUS RC', 'LEXUS RC 300', 'LEXUS RX', 'LEXUS RX 300', 'LEXUS RX 450h', 'MASERATI Ghibli', 'MASERATI GranCabrio', 'MASERATI GranTurismo', 'MASERATI Levante', 'MASERATI The new Quattroporte', 'MAZDA Mazda 2', 'MAZDA Mazda 3', 'MAZDA Mazda BT-50', 'MAZDA Mazda CX-3', 'MAZDA Mazda CX-30', 'MAZDA Mazda CX-5', 'MAZDA Mazda CX-8', 'MERCEDES BENZ C-Class', 'MERCEDES BENZ CLA', 'MERCEDES BENZ E-Class', 'MERCEDES BENZ G-Class', 'MERCEDES BENZ GLA', 'MERCEDES BENZ GLC', 'MERCEDES BENZ GLE', 'MERCEDES BENZ GLS', 'MERCEDES BENZ Maybach', 'MERCEDES BENZ S-Class long', 'MERCEDES BENZ V-Class', 'MG 5', 'MG HS', 'MINI Mini 3 door', 'MINI Mini 5 door', 'MINI Mini Convertible', 'MITSUBISHI All New Xpander', 'MITSUBISHI Attrage', 'MITSUBISHI Mirage', 'MITSUBISHI Outlander', 'MITSUBISHI Pajero Sport', 'MITSUBISHI Triton', 'NISSAN Navara', 'NISSAN Sunny', 'NISSAN Terra', 'NISSAN X-Trail', 'PEUGEOT 3008', 'PEUGEOT 5008', 'PEUGEOT 508', 'PEUGEOT Traveller', 'PORSCHE 718 Boxster', 'PORSCHE 911', 'PORSCHE Gran Turismo Panamera', 'PORSCHE Macan', 'PORSCHE SUV Cayenne', 'PORSCHE coupe 718 Cayman', 'SSANGYONG Rexton', 'SSANGYONG Stavic', 'SSANGYONG Tivoli', 'SUBARU Forester', 'SUBARU Outback', 'SUBARU WRX', 'SUBARU WRX STI', 'SUBARU XV', 'SUZUKI Celerio', 'SUZUKI Ciaz', 'SUZUKI Ertiga', 'SUZUKI Super Carry Pro', 'SUZUKI Super Carry Truck', 'SUZUKI Super Carry Van', 'SUZUKI Swift', 'TOYOTA Alphard', 'TOYOTA Avanza', 'TOYOTA Camry', 'TOYOTA Corolla Cross', 'TOYOTA Fortuner', 'TOYOTA Granvia', 'TOYOTA Hiace', 'TOYOTA Hilux', 'TOYOTA Land Cruiser', 'TOYOTA Land Cruiser Prado', 'TOYOTA Rush', 'TOYOTA Wigo', 'TOYOTA Yaris', 'VINFAST Fadil', 'VINFAST LUX A2.0', 'VINFAST LUX SA2.0', 'VINFAST e34', 'VINFAST vf5', 'VINFAST vf8', 'VINFAST vf9', 'VOLKSWAGEN Beetle Dune', 'VOLKSWAGEN Passat', 'VOLKSWAGEN Polo', 'VOLKSWAGEN Scirocco', 'VOLKSWAGEN Tiguan Allspace', 'VOLKSWAGEN Tiguan Allspace Luxury', 'VOLVO S90 Inscription', 'VOLVO S90 Momentum', 'VOLVO V90 Cross Country', 'VOLVO XC-40 R-Design', 'VOLVO XC60 Inscription', 'VOLVO XC90 Excellence', 'VOLVO XC90 Inscription']
        '''self.class_names = ['ASTON', 'AUDI', 'BAIC', 'BENTLEY', 'BMW', 'BRILLIANCE', 'CHEVROLET', 'FORD', 'HONDA',
                            'HONGQI', 'HYUNDAI', 'ISUZU', 'JAGUAR', 'KIA', 'LAMBORGHINI', 'LAND ROVER', 'LEXUS',
                            'MASERATI', 'MAZDA', 'MERCEDES', 'MG', 'MINI', 'MITSUBISHI', 'NISSAN', 'PEUGEOT', 'PORSCHE',
                            'SSANGYONG', 'SUBARU', 'SUZUKI', 'TOYOTA', 'VINFAST', 'VOLKSWAGEN', 'VOLVO']'''
        self.class_names = ['AUDI', 'FORD', 'HONDA', 'HUYNDAI', 'KIA', 'LEXUS', 'MAZDA', 'MERCEDES', 'MITSUBISHI', 'PEUGEOUT', 'SUZUKI', 'TOYOTA', 'VINFAST', 'VOLVO']
    def classifier(self, image):
        transform = self.data_transforms['test']
        PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
        test_image_tensor = transform(PIL_image)
        if torch.cuda.is_available():
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
        else:
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
        with torch.no_grad():
            self.model.eval()
            # Model outputs log probabilities
            out = self.model(test_image_tensor)
            ps = torch.exp(out)
            topk, topclass = ps.topk(1, dim=1)
            print(out)
            print(ps)
            print(topclass.cpu().numpy()[0][0])
        return self.class_names[topclass.cpu().numpy()[0][0]]


model = CarBrand()
image = Image.open(r"C:\Users\maintn\PycharmProjects\pythonProject\runs\detect\predict13\crops\car\predict13 (33).jpg")
print(model.classifier(image))
