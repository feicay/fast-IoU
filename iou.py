import numpy  as np
from scipy.spatial import ConvexHull
import torch
from torch.autograd import Variable
from numba import jit
import time

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def iou_with_angle(pred, truth):
    """the pred is the predicted box [x, y, w ,h, im, re]
        (x, y) is the centure of the box, (w, h) is the width and the height of the box
        (im, re) means sin(theta), cos(theta), theta is the intersection angle between box and x-axis
    """
    num_p, n = pred.size()
    num_t, n = truth.size()
    assert(n == 6)
    l = pred[:, 2].pow(2).add(pred[:, 3].pow(2)).sqrt() / 2
    alpha = torch.atan2(pred[:, 2], pred[:, 3])
    belta = torch.atan2(pred[:, 4], pred[:, 5])
    x1 = pred[:, 0].add(l.mul(torch.cos(alpha.add(belta)))).view(num_p, 1)
    y1 = pred[:, 1].sub(l.mul(torch.sin(alpha.add(belta)))).view(num_p, 1)
    x2 = pred[:, 0].sub(l.mul(torch.cos(belta.sub(alpha)))).view(num_p, 1)
    y2 = pred[:, 1].add(l.mul(torch.sin(belta.sub(alpha)))).view(num_p, 1)
    x3 = pred[:, 0].sub(l.mul(torch.cos(alpha.add(belta)))).view(num_p, 1)
    y3 = pred[:, 1].add(l.mul(torch.sin(alpha.add(belta)))).view(num_p, 1)
    x4 = pred[:, 0].add(l.mul(torch.cos(belta.sub(alpha)))).view(num_p, 1)
    y4 = pred[:, 1].sub(l.mul(torch.sin(belta.sub(alpha)))).view(num_p, 1)
    pred_point = torch.cat((x1,y1,x2,y2,x3,y3,x4,y4),1).contiguous().view(num_p, 4, 2)
    l = truth[:, 2].pow(2).add(truth[:, 3].pow(2)).sqrt() / 2
    alpha = torch.atan2(truth[:, 2], truth[:, 3])
    belta = torch.atan2(truth[:, 4], truth[:, 5])
    x1 = truth[:, 0].add(l.mul(torch.cos(alpha.add(belta)))).view(num_t, 1)
    y1 = truth[:, 1].sub(l.mul(torch.sin(alpha.add(belta)))).view(num_t, 1)
    x2 = truth[:, 0].sub(l.mul(torch.cos(belta.sub(alpha)))).view(num_t, 1)
    y2 = truth[:, 1].add(l.mul(torch.sin(belta.sub(alpha)))).view(num_t, 1)
    x3 = truth[:, 0].sub(l.mul(torch.cos(alpha.add(belta)))).view(num_t, 1)
    y3 = truth[:, 1].add(l.mul(torch.sin(alpha.add(belta)))).view(num_t, 1)
    x4 = truth[:, 0].add(l.mul(torch.cos(belta.sub(alpha)))).view(num_t, 1)
    y4 = truth[:, 1].sub(l.mul(torch.sin(belta.sub(alpha)))).view(num_t, 1)
    truth_point = torch.cat((x1,y1,x2,y2,x3,y3,x4,y4),1).contiguous().view(num_t, 4, 2)
    pred_point = pred_point.cpu()
    truth_point = truth_point.cpu()
    iou = torch.zeros(num_p)
    idx = torch.zeros(num_p)
    for i in range(num_p):
        pa = pred_point[i,:,:].numpy()
        max_iou = 0
        max_idx = 0
        for j in range(num_t):
            pb = truth_point[j,:,:].numpy()
            if pb[0,0] < 0.000001:
                break
            hull1 = ConvexHull(pa)
            hull2 = ConvexHull(pb)
            point1 = pa[hull1.vertices]
            point2 = pb[hull2.vertices]
            union = hull1.volume + hull2.volume
            inter_point = polygon_clip(point1, point2)
            if inter_point is not None:
                inter_hull = ConvexHull(inter_point)
                interaction = inter_hull.volume
            else:
                interaction = 0.0
            iou_a = interaction / (union - interaction)
            if iou_a > max_iou:
                max_iou = iou_a
                max_idx = j
        iou[i] = max_iou
        idx[i] = max_idx
    return iou, idx

#this is the vectorized box IOU calculation method
def iou_no_angle(pred_box, truth_box):
    num_pred, num_coords = pred_box.size()
    assert(num_coords == 4)
    num_truth, num_coords = truth_box.size()
    assert(num_coords == 4)
    w = Variable(torch.Tensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]))
    zero = Variable(torch.zeros(num_truth))
    iou = Variable(torch.zeros(num_pred))
    if pred_box.is_cuda:
        w = w.cuda()
        zero = zero.cuda()
        iou = iou.cuda()
    truth_box_ex = truth_box.mm(w)
    pred_box_ex = pred_box.mm(w)
    pred_box_ex2 = pred_box_ex.view(1,4*num_pred).expand(num_truth,4*num_pred)
    pred_box_ex3 = pred_box_ex2.view(num_truth,num_pred,4).permute(1,0,2)
    truth_box_ex3 = truth_box_ex.expand(num_pred, num_truth, 4)
    left = torch.max(pred_box_ex3[:,:,0], truth_box_ex3[:,:,0])
    right = torch.min(pred_box_ex3[:,:,2], truth_box_ex3[:,:,2])
    up = torch.max(pred_box_ex3[:,:,1], truth_box_ex3[:,:,1])
    down = torch.min(pred_box_ex3[:,:,3], truth_box_ex3[:,:,3])
    intersection_w = torch.max( right.sub(left), zero)
    intersection_h = torch.max( down.sub(up), zero)
    intersection = intersection_w.mul(intersection_h)
    w_truth = truth_box[:,2].view(1,num_truth).expand(num_pred, num_truth)
    h_truth = truth_box[:,3].view(1,num_truth).expand(num_pred, num_truth)
    w_pred = pred_box[:,2].view(num_pred,1).expand(num_pred, num_truth)
    h_pred = pred_box[:,3].view(num_pred,1).expand(num_pred, num_truth)
    union = torch.add(w_truth.mul(h_truth), w_pred.mul(h_pred)).sub(intersection)
    iou,idx = intersection.div(union).max(dim=1)
    return iou, idx

def test():
    a = np.array([[0.1, 0.1],[0.5,0.5],[0.9,0.2],[0.8,0.4]])
    b = np.array([[0.2,0.2],[0.7,0.3],[0.3,0.7],[0.6,0.5]])
    hull1 = ConvexHull(a)
    hull2 = ConvexHull(b)
    point1 = a[hull1.vertices]
    point2 = b[hull2.vertices]
    union = hull1.volume + hull2.volume
    inter_point = polygon_clip(point1, point2)
    #print(inter_point)
    if inter_point is not None:
        inter_hull = ConvexHull(inter_point)
        interaction = inter_hull.volume
    iou = interaction / (union - interaction)
    #print('iou = %f'%iou)

if __name__ == '__main__':
    truth = torch.Tensor([[0.2,0.2,0.1,0.1,0,1],[0.5,0.5,0.2,0.1,-1,0],[0.3,0.3,0.2,0.2,1,0]])
    pred = torch.Tensor([[0.2,0.2,0.1,0.1,0.7,0.7],[0.5,0.5,0.2,0.1,-0.7,0.7],[0.3,0.3,0.2,0.1,1,0]])
    print(pred.size())
    iou, idx = iou_with_angle(pred, truth)
    print(iou)
    print(idx)
    pred = torch.rand(169*5,4)
    truth = torch.rand(30,4)
    t0 = time.time()
    for i in range(64):
        iou, idx = iou_no_angle(pred, truth)
    t1 = time.time()
    print('cost time %f'%(t1-t0))