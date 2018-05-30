import numpy as np
import math
import random
# import pybinding as pb
# import matplotlib.pyplot as plt
import os
import sys

""" A Lennard Jones Simulation in 2D """
class LJ_2D_Sim_Numpy():
    def __init__(self,sigma,eps,temperature,duration,dt,ip,iv,ncells,adist,rfreq,efreq,rc,vscale):
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
        self.vscale = vscale
    """ Position initialization function """
    def initial_position(self):
        #If condition to initialize lattice if ip is 1
        if self.ip == 1:
            #First of 2 initial atoms to build lattice with
            atom_1 = [0.0,0.0]
            #Second initial atom
            atom_2 = [self.adist/2,self.adist/2]
            #Empty list to contain lattice coordinates
            latticeList = []
            #Nested for loops to build lattice
            for atomx in range(self.ncells):
                for atomy in range(self.ncells):
                    #Formula to create new atom coordinates with initial position @ atom_1
                    newAtom_1 = [atom_1[0] + (self.adist*(atomx)),atom_1[1] + (self.adist*(atomy))]
                    #Formula to create new atom coordinates with initial position @ atom_2
                    newAtom_2 = [atom_2[0] + (self.adist*(atomx)),atom_2[1] + (self.adist*(atomy))]
                    #Appends the new coordinates to the empty list
                    latticeList.append(newAtom_1)
                    latticeList.append(newAtom_2)
            #converts list of coordinates into an array of similar dimensions
            latticeArray = np.array(latticeList)

            return latticeArray
        #If condition to read gro file of lattice coords if ip is 0
        elif self.ip == 0:
            #Empty list that will eventually contain all of the parsed coordinates
            latticeList = []
            #Context manager for input file
            with open('final.gro',"r") as latInput:
                #Parses through each individual line of the file
                lines = latInput.readlines()
                #A for loop to iterate through each line in lines with a counter via the enumerate function
                for count,item in enumerate(lines):
                    #If statement to ensure tha header & box size lines are not considered
                    if count !=0 and count != len(lines)-1:
                        #coords variable to store all of the posX & posY columns
                        coords = [float(item[21:28].strip()),float(item[29:36].strip())]
                        #Appends coords to latticeList
                        latticeList.append(coords)
            #Converts lattice list into an array
            latticeArray = np.array(latticeList)
            return latticeArray
    """ Momentum initialization function """
    def initial_momenta(self,ncells):
        # If fucntion to randomly initialze momenta if iv = 1
        if self.iv == 1:
            u_0 = math.sqrt(2*self.temperature)
            #Empty list that will contain list of momenta for each partcile
            momentaList = []
            #Counter for the total momenta in the x-direction
            totalPX = 0.0 #Is this needed?
            #Counter for the total momenta in the y-direction
            totalPY = 0.0 #Is this needed?
            for atom in range(ncells):
                #Empty list that will contain the momenta of atom
                atomMomenta =[]
                #Generates random theta value
                theta = 2*math.pi*random.random()
                #Produces a random point within the unit circle
                rand_point = [math.cos(theta),math.sin(theta)]
                # point_magnitude = (rand_point[0]**2) + (rand_point[1]**2)
                # sp = math.sqrt((1.0-point_magnitude))*2.0
                PX = rand_point[0] * u_0
                PY = rand_point[1] * u_0
                totalPX += PX
                totalPY += PY
                atomMomenta.append(PX)
                atomMomenta.append(PY)
                momentaList.append(atomMomenta)
            momentaArray = np.array(momentaList)
            return momentaArray
        #if condition to read in initial momenta file
        if self.iv == 0:
            #Empty momentaList which will contain the initial momenta of each atom
            momentaList = []
            #Context manager for input file
            with open('final.gro',"r") as velInput:
                #Reads through each line of the file
                lines = velInput.readlines()
                #For loop to iterate through each line in lines with a counter
                for count,item in enumerate(lines):
                    #If statement to ensure that header and box are not considered
                    if count !=0 and count != len(lines)-1:
                        print(item)
                        print('This is value 1:{} This is value 2:{}'.format(item[37:44].strip(),item[45:52].strip()))
                        coords = [float(item[45:52].strip()),float(item[53:60].strip())]
                        momentaList.append(coords)
            return np.array(momentaList)

    """ Force initialization function """
    def initial_forcesV2(self,lattice):
        #Intialization total potential energy
        ut = 0.0
        #Initialization of Force Array in X & Y direction
        iforcesArrayX = np.zeros((len(lattice),len(lattice)))
        iforcesArrayY = np.zeros((len(lattice),len(lattice)))
        #Main for loop to iterate over every atom
        for atom1 in range(len(lattice)):
            #Nested for loop to iterate over every potential pairwise interaction with atom1
            for atom2 in range(atom1+1,len(lattice)):
                #Calculates distance b/w atom1 and atom2 in x and y direction
                drx = lattice[atom1][0] - lattice[atom2][0]
                dry = lattice[atom1][1] - lattice[atom2][1]
                #Implements periodic boundary condition
                drx -= self.boxsize*round(drx/self.boxsize)
                dry -= self.boxsize*round(dry/self.boxsize)
                #Calculates the magnitude of the calculated distances
                r2 = (drx**2)+(dry**2)
                #If state to ensure that if r2 is within bounds of cutoff range, potential will be calculate based off of LJ potential equation otherwise value is 0.
                if r2 <= self.rc**2:
                    #Initializing components of LJ potential
                    r6r = (1.0)/(r2**3)
                    r12r = r6r ** 2
                    r = math.sqrt(r2)
                    #Calculating LJ potential
                    u = (r12r - r6r)*4.0 - self.urc - (r - self.rc) * self.dudrc
                    #Adding LJ potential to total PE
                    ut += u
                    #Calculating derivative of LJ potential
                    t = 48.0*(r12r - 0.5*r6r)/r2 + (self.dudrc)/r
                    #Initializing each array element with calculated force
                    iforcesArrayX[atom1][atom2] = drx*t
                    iforcesArrayY[atom1][atom2] = dry*t
                #Else statement to initialze array element with force value of 0 if outside of cut off range
                else:
                    iforcesArrayX[atom1][atom2] = 0
                    iforcesArrayY[atom1][atom2] = 0
                #Implements Newton's 3rd law
                iforcesArrayX[atom2][atom1] = -iforcesArrayX[atom1][atom2]
                iforcesArrayY[atom2][atom1] = -iforcesArrayY[atom1][atom2]
        return iforcesArrayX,iforcesArrayY,ut

    """ Sub function to calculate the total force for a given array of forces """
    def totalForce(self,forceX,forceY):
        #Initializes total force Array containing the total forces of each atom in X & Y direction
        tforceArray = np.zeros([self.natoms,2])
        #Atom counter to iterate over each atom
        atomCount = 0
        #For loop to iterate over all atoms
        for atom1 in range(self.natoms):
            #total Force counter for X & Y
            tForceX = 0.0
            tForceY = 0.0
            #second for loop to iterate over all atoms for pair wise interactions
            for atom2 in range(self.natoms):
                #Sums X & Y forces of each pair wise interaction to totalForce
                tForceX += forceX[atom1][atom2]
                tForceY += forceY[atom1][atom2]
            #Equates each tForce to respective atom # in tforceArray
            tforceArray[atomCount][0] = tForceX
            tforceArray[atomCount][1] = tForceY
            #Increases atomcount by one to continue totalforce calculations
            atomCount += 1
        return tforceArray

    """Leap frog Algorithm which takes in position(lattice), force, and momenta of a system and outputs
    the resulting new force & momenta of the system after a given amount of time set by duration."""

    def leapFrogAlgo(self,lattice,forceX,forceY,momenta):
        #Total acceleration in X-direction counter
        accelX = 0.0

        #Total acceleration in Y-direction counter
        accelY = 0.0

        forceList = self.totalForce(forceX,forceY)

        rfreqCounter = 0
        efreqCounter = 0
        #Main for loop which loops through self.duration in intervals of self.dt
        for t in np.arange(0,self.duration,self.dt):
            # print('')
            #Empty list which will store the x_(i+1) positions
            newPosition = []
            #Empty list which will store the v_(i+1) positions
            newVelocity = []
            #Counter for kinetic energy that will reset with each time step
            kineticEnergy = 0.0
            #Counter for potential energy that will reset with each time step
            potentialEnergy = 0.0

            #For loop to sum up the total acceleration of the system
            for i in range(len(forceList)):
                accelX += forceList[i][0]
                accelY += forceList[i][1]

            #Total non-shifted X-acceleration
            daccelX = accelX/float(len(lattice))

            #Total non-shifted Y-acceleration
            daccelY = accelY/float(len(lattice))

            #Nested foor loop whcih will iterate through each atom within the initialized lattice
            for i in range(self.natoms):

                #Leap Frog Algorithm's method for calculating the next position
                newPos = [lattice[i][0] + momenta[i][0]*self.dt + (1/2)*(forceList[i][0]-daccelX)*(self.dt**2),lattice[i][1] + momenta[i][1]*self.dt + (1/2)*(forceList[i][1]-daccelY)*(self.dt**2)]
                #Appends newPos to newPosition list
                newPosition.append(newPos)
                #Calculates the kinetic energy of i'th atom and adds to KE counter
                kineticEnergy += 0.5*((momenta[i][0]**2)+(momenta[i][1]**2))

            #Updates lattice with new Position coordinates
            lattice = newPosition

            #Apply a sub function that will append the results of the completed newPos into a text file
            newForceX,newForceY,ut = self.initial_forcesV2(lattice)
            potentialEnergy += ut
            ntForce = self.totalForce(newForceX,newForceY)

            for i in range(self.natoms):
                #Leap Frog Algorithm's method for calculating the next velocity
                newVel = [momenta[i][0] + (1/2)*(ntForce[i][0]+forceList[i][0]-daccelX)*self.dt, momenta[i][1] + (1/2)*(ntForce[i][1]+forceList[i][1]-daccelY)*self.dt]
                #Appends newVel to the append to the newVelocity list
                newVelocity.append(newVel)
            #If statemetn to determine whether or not to write the file based off of rfreq
            if rfreqCounter%self.rfreq == 0:
                with open(sys.argv[2],"a") as positionTrajectory:
                    positionTrajectory.write('MD sim of {} Argon atoms, t = {}\n'.format(self.natoms,t))
                    positionTrajectory.write('{}\n'.format(self.natoms))
                    for i in range(len(newPosition)):
                        trajOutput ='{resNum:>5d}{resName:>5s}{atomName:>5s}{atomNum:>5d}{posX:8.3f}{posY:8.3f}{posZ:8.3f}\n'.format(resNum = i+1,resName = 'ARGON',atomName ='Ar', atomNum = i+1 ,posX = newPosition[i][0]*self.sigma ,posY =newPosition[i][1]*self.sigma, posZ = 1)
                        positionTrajectory.write(trajOutput)
                    positionTrajectory.write('{0} {0} {0}\n'.format(self.boxsize*self.sigma))
            if efreqCounter%self.efreq == 0:
                with open('te.txt',"a") as teFile,open('pe.txt',"a") as peFile,open('ke.txt',"a") as keFile,open('tFile.txt','a') as tempFile:
                    peFile.write('{time:>3f} {pe:>8.3f}\n'.format(time = t,pe = potentialEnergy))
                    keFile.write('{time:>3f} {ke:>8.3f}\n'.format(time = t,ke = kineticEnergy))
                    teFile.write('{time:>3f} {te:8.3f}\n'.format(time = t,te = kineticEnergy + potentialEnergy))
                    tempFile.write('{time:>3f} {temp:8.3f}\n'.format(time = t, temp = (kineticEnergy*self.eps)/self.natoms))
            #Updates momenta with new velocities
            momenta = newVelocity
            #Updates forceList(a_i) with ntForce(a_i+1) for next iteration
            forceList = ntForce
            rfreqCounter += 1
            efreqCounter += 1
        #Apply a sub funciton that will append the rsults of the completed newVel into a text file
        with open(sys.argv[1],"a") as positionOutput:
            #Writes the header of the final output file
            positionOutput.write('MD Sim {} Argons\n'.format(self.natoms))
            #for loop to iterate through every single newPosition
            for i in range(len(newPosition)):
                #Writes the properly formatted info of the atom and its position&velocity
                posOutput ='{resNum:>5d}{resName:>5s}{atomName:>5s}{atomNum:>5d}{posX:8.3f}{posY:8.3f}{posZ:8.3f}{velX:8.4f}{velY:8.4f}{velZ:8.4f}\n'.format(resNum = i+1,resName = 'ARGON',atomName ='Ar', atomNum = i+1 ,posX = newPosition[i][0] ,posY =newPosition[i][1], posZ = 1,velX =momenta[i][0],velY = momenta[i][1],velZ = 0)
                positionOutput.write(posOutput)
            #Writes the boxsize ender for the final output file
            positionOutput.write('{0} {0} {0}\n'.format(self.boxsize))
        return

    """Plotter function to visualize any given 2D lattice"""
    def Plotter(self,lattice):
        x = lattice[:,0]
        y = lattice[:,1]
        plt.scatter(x,y)
        latplot = plt.show()
        return latplot

def main():
    #(self,sigma,eps,temperature,duration,dt,ip,iv,ncells,adist,rfreq,efreq,rc)
    try:
        os.remove('final.gro')
        os.remove('ke.txt')
        os.remove('pe.txt')
        os.remove('te.txt')
        os.remove('traj.gro')
        os.remove('tFile.txt')
    except IOError:
        pass
    instance = LJ_2D_Sim_Numpy(0.34,120.0,0.5,10,0.001,1,1,16,0.53,1000,10,1.6/1.414,1)
    lattice = instance.initial_position()
    imomenta = instance.initial_momenta(len(lattice))
    if instance.iv == 0 or instance.ip == 0:
        try:
            os.remove('final.gro')
        except FileNotFoundError:
            pass
    iforceX,iforceY,ut = instance.initial_forcesV2(lattice)
    sim = instance.leapFrogAlgo(lattice,iforceX,iforceY,imomenta)

if __name__ == '__main__':
    main()
