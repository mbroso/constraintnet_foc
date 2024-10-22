"""This file summarizes functionality for modelling dynamic constraints for the
class of constraints which are described by polytopes.

The functionality for modelling the dynamic constraints consists namely of: 

    - constr_para2feat: Functors to create constraint features based on 
        constraint parameters. This features are passed to the neural network
        which generates the latent representation. The features inform the 
        neural net about the constraint parameters in an appropriate format.
    - constr_para2repr: Functors to create a representation of constraint 
        parameters based on vertices of the polytopes which describe the 
        constraints in an appropriate format for the constr_mapping function. 
        For the considered class of constraints which are described by 
        polytopes we use vertice based representation:
        
        v_polys (list): 
            [out_part_1/poly_1, out_part_2/poly_2, ...]
            This array describes several polytopes which represent independent 
            constraints for different output parts.
        out_part_i/poly_i (list):
            [v_convex_poly_1, v_convex_poly_2,...]
            ! Number of convex polytopes might differ for different output
            parts. For an output part which is constrained to one convex 
            polytope, the number of convex polytopes is one.
            This array describes one eventually non-convex polytope by 
            specifying a partition into a fixed number of convex polytopes. 
        v_convex_poly_i (torch tensor):
            shape (N, n_v, dim_v)
            The given vertices describe the shape of each convex polytope.
    - constr_mapping: This PyTorch module describes the mapping fct which 
        maps a latent representation which is generated by a special neural 
        network to the output region which is given by the constraint. The 
        constr_mapping function uses as input a latent representation and a 
        constraint rerpresentation. We consider here logits as the latent
        representation and incorportate the softmax functions to the 
        constr_mapping part.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def opts2constr_feat_gen(opts):
    """Creates ConstFeatPlanes functor by calling its constructor with 
    parameters from opts.

    Args:
        opts (obj): Namespace object returned by parser with settings.

    Returns:
        const_feat_planes (obj): Instantiated ConstFeatPlanes functor.
    """
    return ConstrFeatGen(
            opts.const_feat_fac)

class ConstrFeatGen:
    """This constr_para2feat Functor which creates a feature tensor with
    constant channels given by the values of the constraint parameters.

    Each constraint parameter is represented as <repeat> channel(s) with
    specified height and width and constant value.

    """

    def __init__(self, feature_fac):
        """Initialization for setting parameters.

        Args:
            h (int): Height of channels.
            w (int): Width of channels.
            n_channels (int): Number of channels of produced feature tensor.
                Specified number must match
                c_constr_features = n_constr_para * repeat_channels
            repeat_channels (int): The channel for a vertice parameter is
                replicated <repeat> times to amplify its influence.
            norm_factor (float or list): Factor to normalize the values of the
                tensor. If float all constrained parameters are multiplied with
                specified value. If list each constrainted parameters is
                multiplied with corresponding number.
        """
        self.feature_fac = feature_fac

    def __call__(self, constr_para):
        """Functor producing constraint features.

        Args:
            constr_para (obj): Pytorch tensor of shape (N, n_constr_para)
                which specifies the dynamic constraint.

        Returns:
            constr_features (obj): Pytorch tensor representing the constraint
                with shape (N, c_constr_features, H, W) and
        """

        # scale features
        #print(self.feature_fac)
        constr_features = constr_para * self.feature_fac
        return constr_features


def opts2v_polys_acc(opts):
    """Creates VPolysFaceLms functor by calling its constructor with parameters
    from opts.

    Args:
        opts (obj): Namespace object returned by parser with settings.

    Returns:
        v_polys_face_lms (obj): Instantiatet VReprFaceLms functor.
    """
    return VPolysAcc()


class VPolysAcc:
    """This constr_para2repr functor generates a vertex representation in the
    v_polys format for the face landmark prediction in combination with the face
    detector.

    The output (bounding box) of a face detector is used to constrain the
    position of nose and eye landmarks. Furthermore the constraints that the
    eyes are above the nose and that the left eye is more left than the right
    eye are encoded.
    """

    def __init__(self):
        pass

    def __call__(self, constr_para):
        """This function gets constraint parameters and generates the
        corresponding vertices representation.

        Args:
            constr_para (obj): Torch tensor containing the constraint
                parameters with shape (N, n_constr_para). There are 4
                constraint parameters, they are ordered in the following way
                (l_x, u_x, l_y, u_y). They encode the positions of the
                boundaries of the bounding box of the face detector:
                    l_x: left/ lower x
                    u_x: right/ upper x
                    l_y: upper/ lower y (y coordinates start with 0 at the top
                        of the image)
                    u_y: lower/ upper y (y coordinates start with 0 at the top
                        of the image)

        Returns:
            v_polys (obj): Vertice representation of the constraint parameters.
                The output dimensions: (x_nose, x_lefteye, y_righteye,
                y_lefteye, y_righteye, y_nose)
        """
        # 1d polytope for x_nose
        # use constr_para.new to create the tensor on the same device
        # shape (N, n_vertices, dim_vertices), dim_vertices: x_nose
        poly_1d = constr_para.new(constr_para.shape[0], 2, 1)
        # v_1 = (l_x)
        poly_1d[:, 0, 0] = constr_para[:, 0]
        # v_2 = (u_x)
        poly_1d[:, 1, 0] = constr_para[:, 1]

        v_polys = [[poly_1d, ]]

        return v_polys


class ConvexPoly(nn.Module):
    """This nn.Module maps a latent vector in R^N to an output region defined
    by a convex polytope with a fixed number of vertices.

    The shape of this covex polytope is passed as additional input to this 
    module.
    """
    def __init__(self, convex_poly_format):
        """Inform the instance about the expected format of convex polytopes.
        Args:
            convex_poly_format (tuple): Tuple (n_v, dim_v) representing the 
                number of vertices and dimension of the convex polytope.
        """
        super(ConvexPoly, self).__init__()
        self.convex_poly_format = convex_poly_format
        self.dim_z = self.convex_poly_format2dim_z(convex_poly_format)
        self.dim_out = self.convex_poly_format2dim_out(convex_poly_format)

    @staticmethod
    def convex_poly_format2dim_z(convex_poly_format):
        """Extracts the number of latent vector dimensions from 
        convex_poly_format.
        
        Args:
            convex_poly_format (tuple): Tuples (n_v, dim_v) with number of 
                vertices and number of dimensions of convex polytope.
        Returns:
            dim_out (int): Number of output dimensions for given 
                convex_poly_format.
        """
        return convex_poly_format[0]

    @staticmethod
    def convex_poly_format2dim_out(convex_poly_format):
        """Extracts the number of output dimensions from convex_poly_format.

        Args:
            convex_poly_format (tuple): Tuples (n_v, dim_v) with number of 
                vertices and number of dimensions of convex polytope.
        Returns:
            dim_out (int): Number of output dimensions for given 
                convex_poly_format.
        """
        return convex_poly_format[1]

    @staticmethod
    def v_convex_poly2convex_poly_format(v_convex_poly):
        """Extract the convex_poly_format from vertice representation of convex
        polytope.

        Args:
            v_convex_poly (obj): Torch tensor representing the vertices 
                representation of the convex polytope.
        Returns:
            v_convex_poly (tuple): Convex polytope format as tuple of the 
                number of vertices and number of dimensions (n_v, dim_v).
        """
        n_v = v_convex_poly.shape[1]
        dim_v = v_convex_poly.shape[2]
        return (n_v, dim_v)

    def forward(self, z, v_convex_poly):
        """
        Args:
            z (obj): Torch tensor with latent representation. Shape 
                (N, n_v)
            v_convex_poly (obj): Pytorch tensor with convex polytope 
                representation. Shape (N, n_v, dim_v).
        Returns:
            out (obj): Torch tensor with shape (N, dim_v). Each output is 
                within convex polytope specified by v_convex_poly.
        """
        #check convex_poly_format
        obs_convex_poly_format = self.v_convex_poly2convex_poly_format(v_convex_poly)
        if not self.convex_poly_format == obs_convex_poly_format:
            raise TypeError('Expected convex_poly_format does not match \
                    observed one.')

        if not z.shape[1] == self.dim_z:
            raise TypeError('Expected {z_dim} dimensions for latent \
                    representation but observed {z_dim_nn}.'.format(
                        z_dim = self.dim_z,
                        z_dim_nn = z.shape[1])
                    )

        #shape: (N, n_v)
        #print(z.shape)
        p = F.softmax(z, dim=1)
        #change shape to: (N, 1, n_v)
        p = p.view(
                p.shape[0],
                -1,
                p.shape[1]
                )
        #w(N, 1, n_v) * v_convex_poly (N, n_v, dim_v) 
        #= out (N, 1, dim_v)
        out = torch.bmm(p, v_convex_poly)
        #out (N x dim_v)
        out = out.view(out.shape[0], -1)
        
        #check output dimensions
        if not out.shape[1] == self.dim_out:
            raise TypeError('Expected {dim_out} output dimensions but observed \
                    {dim_out_nn}.'.format(
                        dim_out = self.dim_out,
                        dim_out_nn = out.shape[1])
                    )
        return out

class Poly(nn.Module):
    """This nn.Module maps a latent vector to an output region defined polytope 
    which can be defined by several convex polytopes.

    The shape of the non convex polytope is passed by a number of convex 
    polytopes as additional input.
    """
    def __init__(self, poly_format):
        """Information about the encoding of latent and output vector via 
        poly_format.

        Args:
            poly_format (list): List of tuples (n_v, dim_v) for each convex 
                polytope which is part of the total polytope.
        """
        super(Poly, self).__init__()
        #ConvexPoly nn.Module is used for several polytopes
        self.convex_polys = []
        for convex_poly_format in poly_format:
            self.convex_polys.append(ConvexPoly(convex_poly_format))
        self.poly_format = poly_format
        #number of convex polytopes 
        self.n_convex_poly = self.poly_format2n_convex_poly(poly_format)
        #expected dimension of the latent representation
        self.dim_z = self.poly_format2dim_z(poly_format)
        #expected dimensions of the output
        self.dim_out = self.poly_format2dim_out(poly_format)
        if self.n_convex_poly == 0:
            raise TypeError('Polytope must be constructed by at least one \
                    convex polytope.')
        
    @staticmethod
    def poly_format2dim_z(poly_format):
        """Extracts the number of latent vector dimensions from poly_format.
        
        Args:
            poly_format (list): List of tuples (n_v, dim_v) for each convex 
                polytope which is part of the total polytope.
        Returns:
            dim_z (int): Number of latent vector dimensions for given 
                poly_format.
        """
        #dimension of the latent representation
        dim_z = 0
        for convex_poly_format in poly_format:
            dim_z += ConvexPoly.convex_poly_format2dim_z(convex_poly_format)
        #if the polytope is described by more than one convex polytope a 
        #softmax is added and
        n_convex_poly = Poly.poly_format2n_convex_poly(poly_format)
        if n_convex_poly > 1:
            self.dim_z += n_convex_poly

        return dim_z

    @staticmethod
    def poly_format2dim_out(poly_format):
        """Extracts the number of output dimensions from poly_format.

        Args:
            poly_format (list): List of tuples (n_v, dim_v) for each convex 
                polytope which is part of the total polytope.
        Returns:
            dim_out (int): Number of output dimensions for given poly_format.
        """
        #dimensions of the output
        dim_out = 0
        for convex_poly_format in poly_format:
            dim_out += ConvexPoly.convex_poly_format2dim_out(convex_poly_format)
        #if the polytope is described by more than one convex polytope a 
        #softmax is added and 
        n_convex_poly = Poly.poly_format2n_convex_poly(poly_format)
        if n_convex_poly > 1:
            self.dim_out += n_convex_poly
        
        return dim_out
    
    @staticmethod
    def poly_format2n_convex_poly(poly_format):
        """Extracts the number of convex polytopes from poly_format.

        Args:
            poly_format (list): List of tuples (n_v, dim_v) for each convex 
                polytope which is part of the total polytope.
        Returns:
            n_convex_poly^ (int): Number of convex polytopes for given 
                poly_format.
        """
        return len(poly_format)

    @staticmethod
    def v_poly2poly_format(v_poly):
        """Extract the polytope format from vertice description of several 
        convex polytopes.

        Args:
            v_poly (list): List of convex polytope vertice representations 
            which describe an eventually non-convex polytope. v_poly consists
            of torch tensor elements.
        Returns:
            poly_format (list): List of tuples (N, n_v, dim_v) of convex 
            polytopes.
        """
        poly_format = []
        for v_convex_poly in v_poly:
            convex_poly_format = ConvexPoly.v_convex_poly2convex_poly_format(v_convex_poly)
            poly_format.append(convex_poly_format)
        return poly_format



    def forward(self, z, v_poly):
        """
        Args:
            z (obj): Torch tensor with latent representation. Shape (N, dim_z).
                dim_z = sum of number of vertices of convex polytopes + number 
                of convex polytopes when this number is at least two.
            v_poly (list): List of Torch tensors (N, n_v, dim_v) representing 
                convex polytopes. [ (), (), ...]
        Returns:
            out (obj): Torch tensor of shape (N, n_out). n_out = sum of 
                dimensions of each convex polytope + number of convex polytopes 
                when this number is at least two. Format is 
                (p_1, ..., p_K, y_1_1, .. y_1_L, ..., y_k_1, .. y_K_M)
                p_1, ... p_K: probabilities for each convex polytope when 
                number of convex polytopes is greater equal 2. Otherwise these
                probabilities are discarded. y_i_j: Coordinate j within convex 
                polytope i.
        """
        if not z.shape[1] == self.dim_z:
            raise TypeError('Dimension of latent representation in nn is \
                    {dim_z_nn} and required for polytope is {dim_z_poly}. \
                    They should be equal.'.format(
                        dim_z_nn = z.shape[1],
                        dim_z_poly = self.dim_z)
                    )
        
        if not self.poly_format == self.v_poly2poly_format(v_poly):
            raise TypeError('Expectet poly_format, i.e. number of convex \
                    polytopes, number of vertices and their dimensions, does \
                    not match with passed vertices representation of \
                    polytope.')

        #add probabilities for each convex polytope to the output when number
        #of them is greater or equal two.
        out = z.new(z.shape[0], self.dim_out)
        z_current_idx = 0
        out_current_idx = 0
        if self.n_convex_poly > 1:
            #shape: (N, n_convex_poly)
            out = F.softmax(z[:,0:self.n_convex_poly])
            z_current_idx = self.n_convex_poly
            out_current_idx = self.n_convex_poly
        
        for i, convex_poly in enumerate(self.convex_polys):
            v_convex_poly = v_poly[i]
            out[:, out_current_idx: out_current_idx + convex_poly.dim_out] = \
                    convex_poly(
                        z[:, z_current_idx: z_current_idx + convex_poly.dim_z],
                        v_convex_poly)
            z_current_idx += convex_poly.dim_z
            out_current_idx += convex_poly.dim_out

        return out



def opts2polys(opts):
    """Creates Polys nn.Modules by calling its constructor with parameters from 
    opts.

    Args:
        opts (obj): Namespace object returned by parser with settings.

    Returns:
        polys (obj): Instantiated Polys nn.Module.
    """
    #e.g. poly_formats = [[(2,1)],[(3,2)],[(5,3)]]
    #opts.polys_convex_polys_v_n = 2, 3, 5
    #opts.polys_convex_polys_v_dim = 1, 2, 3
    #opts.polys_output_parts = 1, 1, 1
    if not len(opts.polys_convex_polys_v_n) == len(opts.polys_convex_polys_v_dim):
        raise TypeError('Number of list elements in opts.polys_convex_polys_v_n \
                and opts.polys_convex_polys_v_dim must be equal but is not.')
    if not len(opts.polys_convex_polys_v_n) == len(opts.polys_output_parts):
        raise TypeError('Number of list elements in opts.polys_convex_polys_v_n \
                and opts.polys_output_parts must be equal but is not.')
    poly_formats = []
    current_idx = 0
    for n_convex_polys in opts.polys_output_parts:
        poly_format = []
        for i_convex_poly in range(n_convex_polys):
            v_n = opts.polys_convex_polys_v_n[i_convex_poly + current_idx]
            v_dim = opts.polys_convex_polys_v_dim[i_convex_poly + current_idx]
            poly_format.append((v_n, v_dim))
            current_idx += n_convex_polys
        poly_formats.append(poly_format)

    # print('Polys loaded as constr_mapping nn.Module.')

    return Polys(poly_formats)


class Polys(nn.Module):
    """This constr_mapping functor maps a latent vector in R^N to an output 
    which parts are constrained to polytopes.

    Different output parts are constrained to different polytopes 
    independently. These polytopes are passed to this functor as additional
    input in the vertices format v_polys.
    """
    def __init__(self, poly_formats):
        """Inform the Poly object about latent vector dimensions, output 
        dimensions and expected vertice format via poly_formats.

        Args:
            poly_formats (list): List of poly_format objects (see Poly) for 
                the different polytopes of the output parts.
        """
        super(Polys, self).__init__()
        self.poly_formats = poly_formats
        self.polys = []
        for poly_format in self.poly_formats:
            poly = Poly(poly_format)
            self.polys.append(poly)

        #expected number of latent dimensions
        self.dim_z = self.poly_formats2dim_z(poly_formats)
        #expected number of ouput dimensions
        self.dim_out = self.poly_formats2dim_out(poly_formats)

    @staticmethod
    def poly_formats2dim_z(poly_formats):
        """Extracts the number of latent vector dimensions from poly_formats.
        
        Args:
            poly_formats (list): List of poly_format objects (see Poly) for 
                the different polytopes of the output parts.
        Returns:
            dim_z (int): Number of latent vector dimensions for given 
                poly_formats.
        """
        dim_z = 0
        for poly_format in poly_formats:
            dim_z += Poly.poly_format2dim_z(poly_format)
        return dim_z

    @staticmethod
    def poly_formats2dim_out(poly_formats):
        """Extracts the number of output dimensions from poly_formats.

        Args:
            poly_formats (list): List of poly_format objects (see Poly) for 
                the different polytopes of the output parts.
        Returns:
            dim_out (int): Number of output dimensions for given poly_format.
        """
        dim_out = 0
        for poly_format in poly_formats:
            dim_out += Poly.poly_format2dim_out(poly_format)
        return dim_out

    @staticmethod
    def v_polys2poly_formats(v_polys):
        """Extract the polytope formats from vertice description of several 
        polytopes.

        Args:
            v_polys (list): List of polytope vertice representations which 
            describe eventually non-convex polytopes. v_polys consists of list
            elements.
        Returns:
            poly_formats (list): List of poly_format elements.
        """
        poly_formats = []
        for v_poly in v_polys:
            poly_format = Poly.v_poly2poly_format(v_poly)
            poly_formats.append(poly_format)
        return poly_formats

    def forward(self, z, v_polys):
        """
        Args:
            z (obj): Torch tensor with latent representation. Shape (N, n_z).
            v_polys (list): List with polytope description for different output
                parts. 
        Returns:
            out (obj): Torch tensor with 
        """
        #check correct shape of latent representation
        if not z.shape[1] == self.dim_z:
            raise TypeError('Dimension of latent representation is {dim_z_nn}, \
                    but {dim_z} was expected.'.format(
                        dim_z_nn = z.shape[1],
                        dim_z = self.dim_z)
                    )

        #check if v_polys maps with expected poly_formats
        if not self.v_polys2poly_formats(v_polys) == self.poly_formats:
            raise TypeError('Expected format of v_polys given by poly_formats \
                    does not match observed format inferred from v_polys.')

        #output tensor with required dimension
        y = z.new(z.shape[0], self.dim_out)
        
        z_current_idx = 0
        y_current_idx = 0
        for i, poly in enumerate(self.polys):
            dim_z_i = poly.dim_z
            dim_y_i = poly.dim_out
            v_poly = v_polys[i]
            y[:, y_current_idx: y_current_idx + dim_y_i] = \
                    poly(z[:, z_current_idx: z_current_idx + dim_z_i], v_poly)
            z_current_idx += dim_z_i
            y_current_idx += dim_y_i
        return y
       
