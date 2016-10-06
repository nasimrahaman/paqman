__doc__ = """Defines a miniature network inception modules."""


import theano as th
import theano.tensor as T
import numpy as np

import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.archkit as ak
import Antipasti.netools as ntl

# Define shortcuts
# Convlayer with ELU
cl = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                     activation=ntl.elu())

# Convlayer without activation
cll = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize)

# Convlayer with Sigmoid
cls = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                      activation=ntl.sigmoid())

# Strided convlayer with ELU (with autopad)
scl = lambda fmapsin, fmapsout, kersize, padding=None: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout,
                                                                    kersize=kersize,
                                                                    stride=[2, 2], activation=ntl.elu(),
                                                                    padding=padding)

# Strided 3x3 pool layerlayertrain or Antipasti.netarchs.layertrainyard
spl = lambda: nk.poollayer(ds=[3, 3], stride=[2, 2], padding=[1, 1])

# Strided 3x3 mean pool layer
smpl = lambda ds=(2, 2): nk.poollayer(ds=list(ds), poolmode='mean')

# 2x2 Upscale layer
usl = lambda us=(2, 2): nk.upsamplelayer(us=list(us))

# 2x2 Upscale layer with interpolation
iusl = lambda us=(2, 2): nk.upsamplelayer(us=list(us), interpolate=True)

# Batch-norm layer
bn = lambda numinp=None: (nk.batchnormlayer(2, 0.9) if numinp is None else
                          nk.batchnormlayer(2, 0.9, inpshape=[None, numinp, None, None]))

# Softmax
sml = lambda: nk.softmax(dim=2)

# Identity
idl = lambda: ak.idlayer()

# Replicate
repl = lambda numrep: ak.replicatelayer(numrep)

# Merge
merl = lambda numbranch: ak.mergelayer(numbranch)

# Split in half
sptl = lambda splitloc: ak.splitlayer(splits=splitloc, dim=2, issequence=False)

# Dropout layer
drl = lambda p=0.5: nk.noiselayer(noisetype='binomial', p=p)

# Addition layer
adl = lambda numinp: ak.addlayer(numinp, dim=2, issequence=False)

# Circuit layer
crcl = lambda circuit: ak.circuitlayer(circuit, dim=2, issequence=False)

# Parallel tracks
trks = lambda *layers: na.layertrainyard([list(layers)])

lty = lambda ty: na.layertrainyard(ty)


def inceptionize(*towers):
    numtowers = len(towers)
    module = repl(numtowers) + trks(towers) + merl(numtowers)
    return module


def build():
    # Input batch size = (1, 3, 224, 160) after padding. 
    net = inceptionize(scl(3, 16, [9, 9]), scl(3, 16, [5, 5]), scl(3, 16, [3, 3])) + \
          inceptionize(scl(48, 48, [5, 5]), scl(48, 48, [3, 3]), spl()) + \
          inceptionize(scl(144, 48, [5, 5]), scl(48, 48, [3, 3]), spl()) + \
          inceptionize(scl(144, 96, [3, 3]), spl()) + \
          inceptionize(cl(240, 160, [1, 1]) + scl(160, 80, [3, 3]), scl(240, 160, [3, 3]) + cl(160, 80, [1, 1])) + \
          cl(160, 320, [7, 5]) + cl(320, 80, [1, 1]) + cl(80, 40, [1, 1]) + cll(40, 9, [1, 1]) + sml()

    # Define cost fn and optimizer
    net.baggage['learningrate'] = th.shared(value=np.float32(0.0002))
    net.cost(method='cce', regterms=[(2, 0.0005)])
    net.getupdates(method='momsgd', learningrate=net.baggage['learningrate'], nesterov=True)

    return net





