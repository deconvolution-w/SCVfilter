<<<<<<< HEAD
import time
import torch

def train(net, trainloader, testloader, optimizer, criterion, args):
    start = time.time()
    for i in range(args.epoches):
        print(f'Epoch: {i}')
        correct, error = 0, 0
        for x, y in trainloader:
            x, y = x.to(args.device), y.to(args.device)
            pred = net(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_hat = pred.argmax(1)
            correct_num = (y_hat == y).sum().item()
            correct += correct_num
            error_num = (y_hat != y).sum().item()
            error += error_num
        print(f'Train_acc: {correct/(correct+error)*100:.3f}%')
        correct, error = 0, 0
        class_correct = list(0. for i in range(args.nclass))
        class_total = list(0. for i in range(args.nclass))
        for x, y in testloader:
            x, y = x.to(args.device), y.to(args.device)
            pred = net(x)

            y_hat = pred.argmax(1)
            correct_num = (y_hat == y).sum().item()
            correct += correct_num
            error_num = (y_hat != y).sum().item()
            error += error_num
            c = (y_hat == y).squeeze()
            try:
                for j in range(x.size(0)):
                    label = y[j]
                    class_correct[label] += c[j].item()
                    class_total[label] += 1
            except:
                pass
        print(f'Test_acc: {correct / (correct + error) * 100:.3f}%')
        for k in range(args.nclass):
            print('Accuracy of %s: %d %%' % (k, 100 * class_correct[k] / class_total[k]))

        model_path = './models/modelfor' + str(args.nclass) + '_' + str(i) + '.pth'
        torch.save(net, model_path)
        print('Save the model to {}'.format(model_path))
        print('---------------------------------------------------------------------------------------')
=======
import time
import torch

def train(net, trainloader, testloader, optimizer, criterion, args):
    start = time.time()
    for i in range(args.epoches):
        print(f'Epoch: {i}')
        correct, error = 0, 0
        for x, y in trainloader:
            x, y = x.to(args.device), y.to(args.device)
            pred = net(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_hat = pred.argmax(1)
            correct_num = (y_hat == y).sum().item()
            correct += correct_num
            error_num = (y_hat != y).sum().item()
            error += error_num
        print(f'Train_acc: {correct/(correct+error)*100:.3f}%')
        correct, error = 0, 0
        class_correct = list(0. for i in range(args.nclass))
        class_total = list(0. for i in range(args.nclass))
        for x, y in testloader:
            x, y = x.to(args.device), y.to(args.device)
            pred = net(x)

            y_hat = pred.argmax(1)
            correct_num = (y_hat == y).sum().item()
            correct += correct_num
            error_num = (y_hat != y).sum().item()
            error += error_num
            c = (y_hat == y).squeeze()
            try:
                for j in range(x.size(0)):
                    label = y[j]
                    class_correct[label] += c[j].item()
                    class_total[label] += 1
            except:
                pass
        print(f'Test_acc: {correct / (correct + error) * 100:.3f}%')
        for k in range(args.nclass):
            print('Accuracy of %s: %d %%' % (k, 100 * class_correct[k] / class_total[k]))

        model_path = './models/modelfor' + str(args.nclass) + '_' + str(i) + '.pth'
        torch.save(net, model_path)
        print('Save the model to {}'.format(model_path))
        print('---------------------------------------------------------------------------------------')
>>>>>>> 5f5caeb (first commit)
    print(time.time()-start)