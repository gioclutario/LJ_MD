import numpy as np
import math
import random
import pybinding as pb
import matplotlib.pyplot as plt
import sys

""" A Lennard Jones Simulation in 2D """
class LJ_2D_Sim_Numpy():
    def __init__(self,sigma,eps,temperature,duration,dt,ip,iv,ncells,natoms,adist,rfreq,efreq,rc):
        self.sigma = sigma
        self.adist = adist/self.sigma
        self.eps = eps
        self.temperature = temperature
        self.duration = duration
        self.dt = dt
        self.ip = ip
        self.iv = iv
        self.ncells = ncells
        #Box size = ncells * adist
        self.boxsize = ncells * self.adist
        self.natoms = 2*(ncells**2)
        self.iterations = self.duration/self.dt
        self.rfreq = rfreq
        self.efreq = efreq
        self.rc = rc
        #LJ potential
        self.urc = 4.0 * (1.0/(self.rc**12) - 1.0/(self.rc**6))
        #derivative of LJ potential
        self.dudrc = (-48/self.rc) * (1.0/(self.rc**12) - 0.5/(self.rc**6))

    """ Position initialization function """
    def initial_position(self):
        #If condition to initialize lattice if ip is 1
        if self.ip == 1:
            #The reduced lattice distance between atoms in Atomic Units
            reducedLat = self.adist/self.sigma
            #First of 2 initial atoms to build lattice with
            atom_1 = [0.0,0.0]
            #Second initial atom
            atom_2 = [reducedLat/2,reducedLat/2]
            #Empty list to contain lattice coordinates
            latticeList = []
            #Nested for loops to build lattice
            for atomx in range(self.ncells):
                for atomy in range(self.ncells):
                    #Formula to create new atom coordinates with initial position @ atom_1
                    newAtom_1 = [atom_1[0] + (reducedLat*(atomx)),atom_1[1] + (reducedLat*(atomy))]
                    #Formula to create new atom coordinates with initial position @ atom_2
                    newAtom_2 = [atom_2[0] + (reducedLat*(atomx)),atom_2[1] + (reducedLat*(atomy))]
                    #Appends the new coordinates to the empty list
                    latticeList.append(newAtom_1)
                    latticeList.append(newAtom_2)
            #converts list of coordinates into an array of similar dimensions
            latticeArray = np.array(latticeList)

            return latticeArray
        #If condition to read gro file of lattice coords if ip is 0
        # elif self.ip == 0:
        #     with open(sys.argv[1],"r") as gro_file1:
        #
        #     pass

    """ Momentum initialization function """
    def initial_momenta(self,ncells):
        # If fucntion to psuedo-randomly initialze momenta if iv = 1
        if self.iv == 1:
            u_0 = math.sqrt(2*self.temperature)
            #Empty list that will contain list of momenta for each partcile
            momentaList = []
            #Counter for the total momenta in the x-direction
            totalPX = 0.0
            #Counter for the total momenta in the y-direction
            totalPY = 0.0
            for atom in range(ncells):
                #Empty list that will contain the momenta of atom
                atomMomenta =[]
                #While loop that will generate a random momenta for atom
                while True:
                    #Generates random theta value
                    theta = 2*math.pi*random.random()
                    #Produces a random point within the unit circle
                    rand_point = [math.cos(theta),math.sin(theta)]
                    point_magnitude = (rand_point[0]**2) + (rand_point[1]**2)
                    if point_magnitude < 1.0:
                        break
                sp = math.sqrt((1.0-point_magnitude))*2.0
                totalPX += sp * rand_point[0] * u_0
                totalPY += sp * rand_point[1] * u_0
                atomMomenta.append(totalPX)
                atomMomenta.append(totalPY)
                momentaList.append(atomMomenta)
            momentaArray = np.array(momentaList)
            return momentaList
        # if self.iv == 0:
        #     return

    """ Force initialization function """
    # def initial_forces(self,lattice):
    #     size = (len(lattice)*(len(lattice)-1))//2
    #     #becomes pe which then becomes ppe_old
    #     ut = 0.0
    #     iforcesArray = np.empty([size,2])
    #     # iforcesArrayY = np.empty([(lattice*(lattice-1))/2,2])
    #     for atom1 in range(len(lattice)):
    #         for atom2 in range(atom1+1,len(lattice)):
    #             drx = lattice[atom1][0] - lattice[atom2][0]
    #             dry = lattice[atom1][1] - lattice[atom2][1]
    #             r2 = (drx**2) + (dry**2)
    #             if r2 >= (self.rc ** 2)
    #                 r6r = (1.0)/(r2**3)
    #                 r12r = r6r ** 2
    #                 r = math.sqrt(r2)
    #                 u = (r12r - r6r)*4.0 - self.urc - (r - self.rc) * self.dudrc
    #                 ut += u
    #                 t = 48.0 * (r12r - 0.5*r6r)/r2 + (self.dudrc)/r
    #                 for force in range(size):
    #                     iforcesArray[force] =
    #
    #             else:
    #                 iforcesList.append([0.0,0.0])
    #     iforcesArray = np.array(iforcesList)
    #     return iforcesArray

    def initial_forcesV2(self,lattice):
        ut = 0.0
        iforcesArrayX = np.zeros((len(lattice),len(lattice)))
        iforcesArrayY = np.zeros((len(lattice),len(lattice)))
        for atom1 in range(len(lattice)):
            for atom2 in range(atom1+1,len(lattice)):
                drx = lattice[atom1][0] - lattice[atom2][0]
                dry = lattice[atom1][1] - lattice[atom2][1]
                drx -= self.boxsize*round(drx/self.boxsize)
                dry -= self.boxsize*round(drx/self.boxsize)
                r2 = (drx**2)+(dry**2)
                if r2 <= self.rc**2:
                    r6r = (1.0)/(r2**3)
                    r12r = r6r ** 2
                    r = math.sqrt(r2)
                    u = (r12r - r6r)*4.0 - self.urc - (r - self.rc) * self.dudrc
                    ut += u
                    t = 48.0*(r12r - 0.5*r6r)/r2 + (self.dudrc)/r
                    iforcesArrayX[atom1][atom2] = drx*t
                    iforcesArrayY[atom1][atom2] = dry*t
                    iforcesArrayX[atom2][atom1] = drx*t
                    iforcesArrayY[atom2][atom1] = dry*t
                else:
                    iforcesArrayX[atom1][atom2] = 0
                    iforcesArrayY[atom1][atom2] = 0
        return iforcesArrayX,iforcesArrayY

    """ Sub function to calculate the total force for a given array of forces """
    def totalForce(self,forceX,forceY):
        tforceArray = np.zeros([self.natoms,2])
        atomCount = 0
        for atom 1 in range(self.natoms):
            tForceX = 0.0
            tForceY = 0.0
            for atom2 in range(self.atoms):
                tForceX += forceX[atom1][atom2]
                tForceY += forceY[atom1][atom2]
            tforceArray[atomCount][0] = tForceX
            tforceArray[atomCount][1] = tForceY
            atomCount += 1
        return tforceArray

    """Leap frog Algorithm which takes in position(lattice), force, and momenta of a system and outputs
    the resulting new force & momenta of the system after a given amount of time set by duration."""

    def leapFrogAlgo(self,lattice,forceX,forceY,momenta):
        #Kinetic energy counter
        ke = 0.0

        #Total acceleration in X-direction counter
        accelX = 0.0

        #Total acceleration in Y-direction counter
        accelY = 0.0

        #For loop to obtain total acceleration of i'th atom

        forceList = totalForce(forceX,forceY)

        #Total non-shifted X-acceleration
        daccelX = accelX/float(len(lattice))

        #Total non-shifed Y-acceleration
        daccelY = accelY/float(len(lattice))

        #Application of Leap Frog Algorithm; General skeleton for now
        #Main for loop which loops through self.duration in intervals of self.dt
        for t in range(self.dt,self.duration,self.dt):

            #Empty list which will store the x_(i+1) positions
            newPosition = []

            #Empty list which will store the v_(i+1) positions
            newVelocity = []

            #Nested foor loop whcih will iterate through each atom within the initialized lattice
            for i in range(self.natoms):

                #Leap Frog Algorithm's method for calculating the next position
                newPos = [lattice[i][0] + momenta[i][0]*t + 0.5*(forceList[i][0])*(t**2),lattice[i][1] + momenta[i][1]*t + 0.5*(forceList[i][1])*(t**2)]
                newPosition.append(newPos)

            #Apply a sub function that will append the results of the completed newPos into a text file
            newForce = self.initial_forcesV2(newPosition)


            for i in range(self.natoms):
                newVel = [imomenta[i][0]+0.5(newForce[i][0])*t,imomenta[i][1]+0.5(newForce[i][1])*t]
                newVelocity.append(newVel)

            #Apply a sub funciton that will append the rsults of the completed newVel into a text file
        return lfArray

    # Probably don't need this as a function and can just write this on main()
    # def initialize_ofile(self,):
    #     with open(sys.argv[1],'w') as file1, open(sys.argv[2],'w') as file2, open(sys.argv[3],'w') as file3:
    #
    #     pass
    def Plotter(self,lattice):
        x = lattice[:,0]
        y = lattice[:,1]
        plt.scatter(x,y)
        latplot = plt.show()
        return latplot
def main():
    #(self,sigma,eps,temperature,duration,dt,ip,iv,ncells,adist,rfreq,efreq,rc)
    instance = LJ_2D_Sim_Numpy(0.34,120.0,0.5,1.0,0.001,1,1,5,0.53,100,10,1.6/1.414)
    lattice = instance.initial_position()
    print(len(lattice))
    imomenta = instance.initial_momenta(len(lattice))
    iforce = instance.initial_forcesV2(lattice)
    print(iforce)
if __name__ == '__main__':
    main()
