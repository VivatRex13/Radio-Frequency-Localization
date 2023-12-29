import math
import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

### Constants
c = 299792458. # Speed of Light
e = 0.0000001 # Tolerance for Gradient Descent Error
delT = 0.0005 # Time between individual transmissions
sChange = 1. # Time between bursts of transmissions
alpha = 0.0001 # Gradient descent alpha
reps = 100
fontSize = 24

t = 0. # Global Time

class node:
    def __init__(self,xVal,yVal,zVal,tauRX,tauTX,tOffset,tDrift,anchNum,lRX,isLeader):
        # Defining Coordinates of Node
        self.x = xVal
        self.y = yVal
        self.z = zVal

        # Defining Hardware Delays
        self.rx = tauRX
        self.tx = tauTX

        self.offset = tOffset # Defining Offset
        self.offsetEst = 0. # Initializing the Estimate of the Offset

        self.time = tOffset

        self.drift = tDrift
        self.driftEst = math.pow(10,-6)

        self.aNum = anchNum

        if (isLeader):
            self.corr = 0.
        else:
            self.corr = self.offset - lRX + self.rx
        self.corrEst = math.pow(10,-6)
        self.corrMeas = 0.

        self.state = np.matrix([[self.corr],
                                [self.drift]])
        self.stateEst = np.matrix([[self.corrEst],
                                   [self.driftEst]])
        
        self.corred = 0.

        self.N = math.pow((6*math.pow(10,-14)),1)
        self.C = np.matrix([1,0])
        self.A = np.matrix([[1,sChange],
                            [0,1]])
        self.M = np.matrix([[math.pow(10*math.pow(10,-9),2),0],
                            [0,math.pow(math.pow(10,-6),2)]])
        
        self.kfCov = np.matrix([[0.00001,0.],
                                [0.,0.00001]])
        self.k = np.matrix([[0.],
                            [0.]])
        
        self.rGot = 0
        self.lRange = 0
    
    def driftChange(self): # Adjusts the drift
        self.k = self.kfCov * self.C.transpose() / ((self.C * self.kfCov * self.C.transpose())[0,0] + self.N)
        self.kfCov -= self.k * self.C * self.kfCov

        self.state = self.A * self.state + np.matrix([np.random.normal(0,1e-9,1),
                                                      [0]])

        self.stateEst += self.k * (self.corrMeas - (self.C * self.stateEst)[0,0])

        self.kfCov = self.A * self.kfCov * self.A.transpose() + self.M
        self.stateEst = self.A * self.stateEst

        self.corr = self.state[0,0]
        self.offset = self.corr - self.rx + L.rx
        self.corrEst = self.stateEst[0,0]

        self.drift = self.state[1,0]
        self.driftEst = self.stateEst[1,0]
    
    def errDerivX(self,think,distn):
        return((-2 * (dist(self,think) - distn) * (self.x - think.x)) / dist(self,think))
    
    def errDerivY(self,think,distn):
        return((-2 * (dist(self,think) - distn) * (self.y - think.y)) / dist(self,think))
    
    def errDerivZ(self,think,distn):
        return((-2 * (dist(self,think) - distn) * (self.z - think.z)) / dist(self,think))
    
    def errDerivTX(self,other,think,tdoa):
        return((2 * (dist(self,think) - dist(other,think) - tdoa)) * ((think.x - self.x) / (dist(self,think)) - (think.x - other.x) / (dist(other,think))))
    
    def errDerivTY(self,other,think,tdoa):
        return((2 * (dist(self,think) - dist(other,think) - tdoa)) * ((think.y - self.y) / (dist(self,think)) - (think.y - other.y) / (dist(other,think))))
    
    def errDerivTZ(self,other,think,tdoa):
        return((2 * (dist(self,think) - dist(other,think) - tdoa)) * ((think.z - self.z) / (dist(self,think)) - (think.z - other.z) / (dist(other,think))))

def dist(node1,node2): # Finds the distance between two nodes
    return(math.sqrt(math.pow(node1.x-node2.x,2) + math.pow(node1.y-node2.y,2) + math.pow(node1.z-node2.z,2)))

def tDist(node1,node2):
    return(dist(node1,node2) / c)

def randRTX(): # Finds a random value for taurx and tautx
    return(random.randint(100,300) * math.pow(10,-8))

def randOffset(): # Finds a random value for tOffset
    return(random.randint(-1000,1000) * math.pow(10,-4))

def tChange(): # Advances time
    global t
    global R
    t += delT

    for i in followers:
        i.time += delT
    
    R.time += delT

def stateChange():
    global t
    global R
    tAdv = sChange - delT * (2 * (len(followers) + 1))
    t += tAdv

    for i in followers:
        i.driftChange()
        i.time += t + i.offset
    
    R.driftChange()
    R.time += t + R.offset

def gradDesc(nodes,leader,tol):
    global alpha

    err = 10.
    errDX = 0.
    errDY = 0.
    errDZ = 0.
    prevErr = 0.

    xT = 0.
    yT = 0.
    zT = 0.

    thinkNode = node(xT,yT,zT,0,0,0,0,-1,0,False)

    while (abs(err-prevErr)>tol):
        prevErr = err
        err = math.pow(dist(leader,thinkNode) - rlDist,2)
        errDX = leader.errDerivX(thinkNode,rlDist)
        errDY = leader.errDerivY(thinkNode,rlDist)
        errDZ = leader.errDerivZ(thinkNode,rlDist)

        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                err += math.pow(dist(nodes[i],thinkNode)-dist(nodes[j],thinkNode)-c*(nodes[i].corred-nodes[j].corred),2)
                errDX += nodes[i].errDerivTX(nodes[j],thinkNode,(nodes[i].corred-nodes[j].corred)*c)
                errDY += nodes[i].errDerivTY(nodes[j],thinkNode,(nodes[i].corred-nodes[j].corred)*c)
                errDZ += nodes[i].errDerivTZ(nodes[j],thinkNode,(nodes[i].corred-nodes[j].corred)*c)
        
        thinkNode.x -= alpha*errDX
        thinkNode.y -= alpha*errDY
        thinkNode.z -= alpha*errDZ

    return(thinkNode)
random.seed(12)
L = node(10.,10.,10.,randRTX(),randRTX(),0.,0.,-1,0,True) # Leader node

R = node(1.,5.,-2.,randRTX(),randRTX(),randOffset(),0.,-1,L.rx,False) # Roaming node

# Followers
numNodes = 4
f1 = node(17.,14.,12.,randRTX(),randRTX(),randOffset(),0.000001,0,L.rx,False)
f2 = node(9.,-7.,-10.,randRTX(),randRTX(),randOffset(),0.000001,1,L.rx,False)
f3 = node(-8.,4.,-12.,randRTX(),randRTX(),randOffset(),0.000001,2,L.rx,False)
f4 = node(-9.,-13.,10.,randRTX(),randRTX(),randOffset(),0.000001,3,L.rx,False)

rep = 0
totErr = 0
corrDev = 0
driftDev = 0
errDev = 0

tP = []
corrP = []
corrEstP = []
driftP = []
driftEstP = []
corrErrP = []
driftErrP = []
errorP = []

followers = [f1,f2,f3,f4]

while (rep < reps):
    ranging = []

    for i in followers:
        ranging.append([])

    ranging.append([])

    for i in range(len(followers)):
        ranging[i].append(t + followers[i].tx + followers[i].rx + followers[i].offset)
        ranging[i].append(t + followers[i].tx + tDist(followers[i],L) + L.rx + L.offset)

        tChange()

        ranging[i].append(t + L.tx + L.rx + L.offset)
        ranging[i].append(t + L.tx + tDist(L,followers[i]) + followers[i].rx + followers[i].offset)

        tChange()
    
    ranging[len(followers)].append(t + R.tx + R.rx + R.offset)
    ranging[len(followers)].append(t + R.tx + tDist(R,L) + L.rx + L.offset)

    for i in range(len(followers)):
        ranging[i].append(t + R.tx + tDist(R,followers[i]) + followers[i].rx + followers[i].offset)

    tChange()

    ranging[len(followers)].append(t + L.tx + L.rx + L.offset)
    ranging[len(followers)].append(t + L.tx + tDist(L,R) + R.rx + R.offset)

    tChange()

    for i in range(len(followers)):
        followers[i].corrMeas = (ranging[i][0]+ranging[i][3]-ranging[i][1]-ranging[i][2]) / 2
        followers[i].lRange = ranging[i][3]
    
    rl_rx0 = ranging[len(followers)][0]
    lr_rx0 = ranging[len(followers)][1]
    lr_rx1 = ranging[len(followers)][2]
    rl_rx1 = ranging[len(followers)][3]

    for i in range(len(followers)):
        followers[i].rGot = ranging[i][4]
    
    rlTauP = (lr_rx0 + rl_rx1 - rl_rx0 - lr_rx1) / 2
    rlDist = rlTauP*c

    stateChange()

    for i in followers:
        i.corred = i.rGot - (i.corrEst + i.driftEst * (i.rGot - i.lRange))

    nodeThink = gradDesc(followers,L,e)
    
    tP.append(t)
    corrP.append(f1.corr*1e6)
    corrEstP.append(f1.corrEst*1e6)
    driftP.append(f1.drift*1e6)
    driftEstP.append(f1.driftEst*1e6)
    corrErrP.append((f1.corrEst-f1.corr)*1e9)
    driftErrP.append((f1.driftEst-f1.drift)*1e9)
    errorP.append(dist(nodeThink,R))
    totErr += dist(nodeThink,R)
    corrDev += math.pow(f1.corrEst-f1.corr,2)
    driftDev += math.pow(f1.driftEst-f1.drift,2)

    print("Location: ", nodeThink.x, ",", nodeThink.y, ",", nodeThink.z)
    print("Real Location: ", R.x, ",", R.y, ",", R.z)
    print("Error: ", dist(nodeThink,R))
    print()

    rep += 1

avgErr = totErr/reps
avgErrP = []
avgRep = 0
while (avgRep < reps):
    avgErrP.append(avgErr)
    avgRep += 1
for x in errorP:
    errDev += math.pow(x,2)

corrStdDev = math.sqrt(corrDev/(reps-1))
driftStdDev = math.sqrt(driftDev/(reps-1))
errStdDev = math.sqrt(errDev/(reps-1))

print("Average Distance Error:", avgErr)
print("Standard Deviation of Distance Error:", errStdDev)
print("Standard Deviation of Correction Error:", corrStdDev)
print("Standard Deviation of Drift Error:", driftStdDev)

plt.figure()
plt.plot(tP,corrP,label="Ground Truth",linewidth=4)
plt.plot(tP,corrEstP,"-.",label="Estimate",linewidth=4)
plt.title("Follower 1 Correction Estimate and Ground Truth vs Time",fontweight="bold",fontsize=fontSize)
plt.ylabel("Correction (microseconds)",fontsize=fontSize)
plt.xlabel("Time (seconds)",fontsize=fontSize)
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(tP,driftP,label="Ground Truth",linewidth=4)
plt.plot(tP,driftEstP,"-.",label="Estimate",linewidth=4)
plt.title("Follower 1 Drift Estimate and Ground Truth vs Time",fontweight="bold",fontsize=fontSize)
plt.ylabel("Drift (microseconds/second)",fontsize=fontSize)
plt.xlabel("Time (seconds)",fontsize=fontSize)
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(tP[5::],corrErrP[5::],linewidth=4)
plt.title("Follower 1 Correction Estimate Error vs Time",fontweight="bold",fontsize=fontSize)
plt.ylabel("Correction Error (nanoseconds)",fontsize=fontSize)
plt.xlabel("Time (seconds)",fontsize=fontSize)
plt.grid(True)

plt.figure()
plt.plot(tP[5::],driftErrP[5::],linewidth=4)
plt.title("Follower 1 Drift Estimate Error vs Time",fontweight="bold",fontsize=fontSize)
plt.ylabel("Drift Error (nanoseconds/second)",fontsize=fontSize)
plt.xlabel("Time (seconds)",fontsize=fontSize)
plt.grid(True)

plt.figure()
plt.plot(tP,errorP,label="Error",linewidth=4)
plt.plot(tP,avgErrP,label="Average Error",linewidth=4)
plt.title("Follower 1 Position Error vs Time",fontweight="bold",fontsize=fontSize)
plt.ylabel("Location Error (meters)",fontsize=fontSize)
plt.xlabel("Time (seconds)",fontsize=fontSize)
plt.legend()
plt.grid(True)

plt.show()