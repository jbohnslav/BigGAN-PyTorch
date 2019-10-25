import os
import numpy as np
import torch
import time
import inception_utils
import utils
import cv2
from queue import LifoQueue, Queue, Empty
from threading import Thread, Lock
import random
from scipy.interpolate import LinearNDInterpolator, griddata, interp1d
import sys
sys.setrecursionlimit(1500)

queue = Queue(8)

should_save = False
category = 607
# category = 298
batch_size = 64
noise_variance = 1
truncate_val = 2

imsize=720
strategy = 'ewma_grad'
smooth_style = 'cubic'
smooth_val = 16
reset_every = 10
step_size = 0.1 # in measures of standard deviations [0, ~0.2]

debug = False
smooth_across_batches = True
interpolation = cv2.INTER_CUBIC
name = 'random_walk'


dim_z = 120
# dim_z = 2

# good combinations
# RANDOM_WALK + cubic smoothing + smooth: 16, reset_every 10, smooth across batches
# ewma_grad + cubic smoothing + smooth: 16, reset every 10, smooth across batches

class Sampler:
    def __init__(self, strategy, z_mean=0, z_var=noise_variance, size=dim_z, ndim=2, 
      scratch=False, reset_every=reset_every, smooth_style=smooth_style, truncate_val=truncate_val,
      smooth_across_batches=smooth_across_batches):
        self.z_mean = z_mean
        self.z_var = z_var
        # self.shape = (batch_size, ndim)
        self.size= size
        self.scratch = scratch
        self.reset_every = reset_every
        self.batch_num = 0
        self.smooth_across_batches = smooth_across_batches
        # self.ndim = ndim

        assert(strategy in ['random', 'random_walk', 'ewma_grad', 'ewma_position', 
          'biased_random'])
        if strategy == 'random':
            self.sample = self.random
        elif strategy == 'random_walk':
            self.sample = self.random_walk
        elif strategy == 'ewma_grad':
            self.sample = self.ewma_grad
        elif strategy == 'ewma_position':
            self.sample = self.ewma_position
        elif strategy == 'biased_random':
            self.sample = self.biased_random
        else:
            raise ValueError('unk value of sampler: {}'.format(strategy))
            
        if smooth_style is not None:
            assert(smooth_style in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'])
            self.smooth_style = smooth_style
            self.should_smooth = False
        else:
            self.should_smooth = False
        self.truncate_val = truncate_val
            
        
    def random(self, n=1):
        return(np.random.normal(loc=self.z_mean, 
                               scale=self.z_var, size=(self.size, n)))
    
    def random_walk(self, n=1,  step_size=step_size):
        samples = []
        if (self.batch_num % self.reset_every) ==0 and hasattr(self, 'prev_sample') :
            # print('restart')
            samples = self.reset()
        else:
            for i in range(n):
                if not hasattr(self, 'prev_sample'):
                    print('scratch')
                    sample = np.random.normal(loc=self.z_mean, 
                                   scale=self.z_var, size=(self.size, 1))
                else:
                    sample = np.random.normal(loc=self.prev_sample, 
                                   scale=step_size, size=(self.size, 1))
                sample = self.truncate(sample)
                self.prev_sample = sample.copy()
                samples.append(sample)
            samples = np.concatenate(samples,axis=1)
            if self.should_smooth:
                samples = self.smooth(samples)
        self.batch_size = n
        self.batch_num +=1
            # print(samples.shape)
        self.prev_batch = samples.copy()
        return(samples)
    
    def ewma_grad(self, n=1, weight=0.9):
        samples = []
        if (self.batch_num % self.reset_every) ==0 and hasattr(self, 'prev_sample') :
          print('restart')
          samples = self.reset()
        else:
          for i in range(n):
              if i==0 and not hasattr(self, 'prev_sample'):
                  print('scratch')
                  sample = np.random.normal(loc=self.z_mean, 
                                 scale=self.z_var, size=(self.size, 1))
              elif i==1 and not hasattr(self, 'prev_grad'):
                  prev_s = self.prev_sample
                  new = np.random.normal(loc=self.z_mean, 
                                 scale=self.z_var, size=(self.size, 1))
                  grad = new - prev_s
                  sample = prev_s + grad
                  self.prev_grad = grad.copy()
              elif hasattr(self, 'prev_grad'):
                  prev_s = self.prev_sample
                  new = np.random.normal(loc=self.z_mean, 
                                 scale=self.z_var, size=(self.size, 1))
                  grad = new - prev_s
                  sample = prev_s + (1-weight)*grad + (weight)*self.prev_grad
                  self.prev_grad = grad.copy()
              else:
                  raise ValueError('not supposed to get here')
              try:
                  sample = self.truncate(sample)
              except RecursionError:
                  prev_s = self.prev_sample
                  new = np.random.normal(loc=self.z_mean, 
                                 scale=self.z_var/10, size=(self.size, 1))
                  grad = new - prev_s
                  sample = prev_s + (0.9)*grad + (0.1)*self.prev_grad
                  self.prev_grad = grad.copy()
                  print('recursion error')

              self.prev_sample = sample.copy()
              samples.append(sample)
          samples = np.concatenate(samples,axis=1)
          if self.should_smooth:
            samples = self.smooth(samples)
        self.batch_num +=1 
        self.batch_size = n
        return(samples)
    
    def ewma_position(self, n=1, weight=0.8):
        samples = []
        # print(self.batch_num % self.reset_every)
        # restart = self.scratch or not hasattr(self, 'prev_sample') or (self.batch_num % self.reset_every) ==0
        if (self.batch_num % self.reset_every) ==0 and hasattr(self, 'prev_sample') :
          # print('restart')
          samples = self.reset()
        else:
          for i in range(n):
              if not hasattr(self, 'prev_sample'):
                  print('scratch')
                  sample = np.random.normal(loc=self.z_mean, 
                                 scale=self.z_var, size=(self.size, 1))
              else:
                  prev_s = self.prev_sample
                  new = np.random.normal(loc=0, 
                                   scale=self.z_var, size=(self.size, 1))
                  sample =  weight*prev_s + (1-weight)*new
              sample = self.truncate(sample)
              samples.append(sample)
              self.prev_sample = sample.copy()
          samples = np.concatenate(samples,axis=1)
          if self.should_smooth:
            samples = self.smooth(samples)
        self.batch_num +=1 
        self.batch_size = n
        # print(self.prev_sample.shape)
        return(samples)
    
    def biased_random(self, n=1, step_size=0.1, weight=0.9):
        if (self.batch_num % self.reset_every) ==0 and hasattr(self, 'prev_sample') :
            # print('restart')
            samples = self.reset()
        else:
            samples = []
            for i in range(n):
                if i==0 and not hasattr(self, 'prev_sample'):
                    sample = np.random.normal(loc=self.z_mean, 
                                   scale=step_size, size=(self.size, 1))
                elif i ==1 or (i==0 and hasattr(self, 'prev_sample')):
                    new = np.random.normal(loc=self.prev_sample, 
                                   scale=step_size, size=(self.size, 1))
                    # sample =  weight*prev_s + (1-weight)*new
                    grad = new - self.prev_sample
                    sample = self.prev_sample + grad
                    prev_grad = grad.copy()
                elif i >1:
                    new = np.random.normal(loc=self.prev_sample, 
                                   scale=step_size, size=(self.size, 1))
                    grad = new - self.prev_sample
                    sample = self.prev_sample + weight*grad + (1-weight)*prev_grad
                    prev_grad = grad.copy()
                sample = self.truncate(sample)
                self.prev_sample = sample.copy()
                samples.append(sample)

            samples = np.concatenate(samples,axis=1)
            if self.should_smooth:
              samples = self.smooth(samples)
        self.batch_num +=1 
        self.batch_size = n

        return(samples)
    def truncate(self, vec):
      # print(vec)
      if self.truncate_val is None:
        return(vec)
      to_trunc = np.where(vec>self.truncate_val)[0]
      # print(len(to_trunc))
      if len(to_trunc)==0:
          pass
          # print(vec.shape)
          # return(vec)
      else:
          # print(to_trunc)
          newvals = np.random.normal(loc=self.z_mean, 
                                 scale=self.z_var, size=(len(to_trunc), ))
          # print(newvals.shape)
          # print(vec.shape)
          # print(to_trunc.shape)
          vec[to_trunc,0] = newvals
          vec = self.truncate(vec)
      return(vec)
    
    def smooth(self, samples, downsample=smooth_val):
        # print('smoothing')
        ndim, npoints = samples.shape
        points = np.linspace(0,1,npoints)
        sampled_points = np.linspace(0,1,npoints//downsample)
        new_samples = []
        for i in range(samples.shape[0]):
            f = interp1d(sampled_points, samples[i,::downsample], kind=self.smooth_style)
            y_hat = f(points)
            # y_hat = np.interp(points, sampled_points, samples[i, ::downsample])
            new_samples.append(y_hat)

        new_samples = np.stack(new_samples)
        # print(new_samples.shape)
        self.prev_sample = new_samples[:,-1:].copy()
        return(new_samples)
    
    def reset(self):
        if not hasattr(self, 'prev_sample'):
            raise ValueError('can''t call reset before a batch')
        x0 = self.prev_sample
        x1 = np.random.normal(loc=self.z_mean, 
                           scale=self.z_var, size=(self.size, 1))
        new_samples = []
        x = np.array([0,1])
        # print(x0.shape)
        newx = np.linspace(0,1,self.batch_size)
        # print(x.shape)
        # print(x0.shape)
        # print(x1.shape)
        for i in range(self.size):
            arr = np.array([x0[i], x1[i]]).squeeze()
            # print(arr.shape)
            f = interp1d(x, arr, kind='linear')
            y_hat = f(newx)
            new_samples.append(y_hat)
            if i > 0:
              self.prev_grad = new_samples[i] - new_samples[i-1]
            # print(self.prev_sample.shape)
        new_samples = np.stack(new_samples)
        self.prev_sample = new_samples[:,-1:]
        # print(self.prev_sample)
        # print(self.prev_sample.shape)
        return(new_samples)
    def smooth_two_batches(self, b0, b1):
        # print(b0.shape, b1.shape)
        big_batch = np.concatenate((b0,b1), axis=1)
        smoothed = self.smooth(big_batch)
        b1 = smoothed[:, b0.shape[1]:]
        return(b1)
    def smooth_n_batches(self, batches):
        batch_size = batches[0].shape[1]
        n = len(batches)
        big_batch = np.concatenate(batches, axis=1)
        smoothed = self.smooth(big_batch)
        out = []
        for i in range(n):
          out.append(smoothed[:, i*batch_size:i*batch_size+batch_size])
        return(out)


# class ImageProducer(Thread):
#   def __init__(self, G, y_, z_, batch_size=32):
#     super().__init__()
#     self.G = G
#     self.y_ = y_
#     self.z_ = z_
#     self.batch_size=batch_size
#     self.sampler = Sampler()
#     self.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))

#   def stop(self):
#     self.should_continue = False

#   def run(self):
#     global queue
#     self.should_continue = True
#     while self.should_continue:
#       if not queue.full():
#         with torch.no_grad():
#           z = torch.Tensor(self.sampler.sample().astype(np.float32)).to(device)
#           # self.z_.sample_()
#           if not hasattr(self, 'x0'):
#             self.x0 = self.z_[0,:].clone()
#             self.x1 = self.z_[-1,:].clone()
#           else:
#             # z_.sample_()
#             self.x0 = self.x1.clone()
#             self.x1 = self.z_.clone()[-1,:]
#           ims = generate_images(self.G, self.y_, self.x0,self.x1, batch_size=self.batch_size)
#         # for im in ims:
#           queue.put(ims)
#           del(z)
#           del(ims)

class ImageProducer(Thread):
  def __init__(self, G, y_, z_, batch_size=32):
    super().__init__()
    self.G = G
    self.y_ = y_
    self.z_ = z_
    self.batch_size=batch_size
    self.sampler = Sampler(strategy)
    self.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    # self.total_n =0 
  def stop(self):
    self.should_continue = False

  def run(self):
    global queue
    self.should_continue = True
    while self.should_continue:
      if not queue.full():
        with torch.no_grad():
          try:
            zs = []
            for i in range(reset_every):
              zs.append(self.sampler.sample(n=self.batch_size).astype(np.float32))
            zs = self.sampler.smooth_n_batches(zs)
            # for i in range(batch_size):
            #   z.append(self.sampler.sample().astype(np.float32))
            # z = np.concatenate(z, axis=1).T
            for z in zs:
              z = torch.Tensor(z.T).to(self.device)
              ims = generate_images_from_z(self.G, self.y_, z, batch_size=self.batch_size)
              queue.put(ims)
              del(z)
              del(ims)
          except KeyboardInterrupt:
            raise

def generate_images_from_z(G, y_, z, batch_size=32, category=category, imsize=imsize, interpolation=interpolation):
    try:
      with torch.no_grad():
          # tmp_z = utils.interp(x0, x1, batch_size-2).squeeze()
          y_.fill_(category)
          # y_.sample_()
          # print(y_)
          # print(z.shape)
          o = G(z, G.shared(y_))
          o = o.detach().cpu().numpy()
      ims = [o[i].transpose(1,2,0) for i in range(o.shape[0])]
      for i in range(len(ims)):
        im = ims[i]
        im = ((im+1)/2).clip(min=0, max=1)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (imsize,imsize), interpolation=interpolation)
        ims[i] = im
    except KeyboardInterrupt:
      raise
    return(ims)


def generate_images(G, y_, x0, x1, batch_size=32, category=category, imsize=imsize):
    with torch.no_grad():
        tmp_z = utils.interp(x0, x1, batch_size-2).squeeze()
        y_.fill_(category)
        # y_.sample_()
        # print(y_)
        o = G(tmp_z, G.shared(y_))
        o = o.detach().cpu().numpy()
    ims = [o[i].transpose(1,2,0) for i in range(o.shape[0])]
    for i in range(len(ims)):
      im = ims[i]
      im = ((im+1)/2).clip(min=0, max=1)
      im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
      im = cv2.resize(im, (imsize,imsize), cv2.INTER_CUBIC)
      ims[i] = im
    return(ims)


def run(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
                
  # Optionally, get the configuration from the state dict. This allows for
  # recovery of the config provided only a state dict and experiment name,
  # and can be convenient for writing less verbose sample shell scripts.
  if config['config_from_name']:
    utils.load_weights(None, None, state_dict, config['weights_root'], 
                       config['experiment_name'], config['load_weights'], None,
                       strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
      if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
        config[item] = state_dict['config'][item]
  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])
   
  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  # print('Experiment name is %s' % experiment_name)
  
  G = model.Generator(**config).cuda()
  utils.count_parameters(G)
  
  # Load weights
  # print('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=False, load_optim=False)
  # Update batch size setting used for G
  G_batch_size = max(config['G_batch_size'], config['batch_size']) 
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'], 
                             z_var=config['z_var'])
  # print(z_.shape)
  G.eval()
    
  p = ImageProducer(G, y_, z_, batch_size=G_batch_size)
  p.daemon=True
  print('starting...')
  p.start()
  time.sleep(1)

  cv2.namedWindow('PumGAN', cv2.WINDOW_AUTOSIZE)
  should_continue = True
  fps = 30
  total_n = 0 

  while should_continue:
    if queue.empty():
      # print('empty')
      time.sleep(1/fps)
      continue
    ims = queue.get()
    queue.task_done()
    # print(queue.qsize())
    for im in ims:
      if should_save:
        if 'writer' not in locals():
          print('SAVING TO DISK!')
          fourcc = cv2.VideoWriter_fourcc(*'MJPG')
          outfile = r'C:\Users\jbohn\Desktop\\' + name + '.avi'
          writer = cv2.VideoWriter(outfile, fourcc, float(fps), (im.shape[1], im.shape[0]))
        # print(im.min())
        # print(im.max())
        # print(im.shape)
        # should_continue=False
        # break
        writer.write((im*255).clip(min=0, max=255).astype(np.uint8))
      if debug:
        cv2.putText(im, '{:04d}'.format(total_n), (10, imsize-25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
          lineType=cv2.LINE_AA)
      cv2.imshow('PumGAN', im)
      total_n+=1
      k = cv2.waitKey(int(1000/fps))
      if k==27:
        should_continue = False
        break
  p.stop()
  cv2.destroyAllWindows()
  if should_save:
    writer.release()
  print('Finished')


def main():
  # parse command line and run    
  parser = utils.prepare_parser()
  parser = utils.add_sample_parser(parser)
  args = parser.parse_args()
  args.experiment_name = '138k'
  args.G_attn = '64'
  args.D_attn = '64'
  args.shuffle = True
  args.num_G_accumulations = 8
  args.num_D_accumulations = 8
  args.num_D_steps = 1
  args.G_lr = 1e-4
  args.D_lr = 1e-4
  args.D_B2 = 0.999
  args.G_B2 = 0.999

  args.hier = True
  args.G_ch = 96
  args.D_ch = 96
  args.G_nl = 'inplace_relu'
  args.D_nl = 'inplace_relu'
  args.SN_eps = 1e-6
  args.BN_eps = 1e-5
  args.adam_eps = 1e-6
  args.G_ortho = 0.0
  args.G_shared = True
  args.G_init = 'ortho'
  args.D_init = 'ortho'
  args.dim_z = 120
  args.shared_dim = 128

  args.G_eval_mode = True
  args.ema = True
  args.use_ema = True
  args.ema_start = 20000
  args.test_every = 2000
  args.save_every = 1000
  args.num_best_copies = 5
  args.num_save_copies = 2
  args.use_multiepoch_sampler = True
  args.batch_size = batch_size

  args.sample_sheet = True
  args.z_var = noise_variance

  config = vars(args)
  # print(config)
  run(config)
  
if __name__ == '__main__':    
  main()