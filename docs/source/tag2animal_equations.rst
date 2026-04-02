Tag-to-Animal Orientation Correction: Equations
=================================================

This page documents the mathematical equations used in
``pyologger.calibrate_data.tag2animal.orientation_and_heading_correction``
to rotate tag sensor data into the animal's reference frame and compute
pitch, roll, and heading.

Step 1 — Normalize the stationary accelerometer vector
------------------------------------------------------

.. math::

   \bar{a} = \frac{\bar{a}_0}{\|\bar{a}_0\|}

Step 2 — Initial pitch and roll (tag-to-animal offset)
------------------------------------------------------

Compute the initial pitch :math:`p_0` and roll :math:`r_0` that describe
the tag's mounting offset relative to the animal, assuming the animal is
stationary on its belly (:math:`p_0 = 0`, :math:`r_0 = 0` is the ideal
aligned case):

.. math::

   p_0 = -\arcsin\!\left(\bar{a}_x\right)

.. math::

   r_0 = \arctan2\!\left(\bar{a}_y,\; \bar{a}_z\right)

With the constraint that :math:`p_0 \in [-\tfrac{\pi}{2}, \tfrac{\pi}{2}]`.
If :math:`p_0 > \tfrac{\pi}{2}`, apply:

.. math::

   p_0 \leftarrow \tfrac{\pi}{2} - p_0, \qquad r_0 \leftarrow r_0 + \pi

Step 3 — Rotation matrices
--------------------------

**Pitch** (rotation about the :math:`y`-axis):

.. math::

   R_P(p) =
   \begin{bmatrix}
     \cos p & 0 & \sin p \\
     0      & 1 & 0      \\
     -\sin p & 0 & \cos p
   \end{bmatrix}

**Roll** (rotation about the :math:`x`-axis):

.. math::

   R_R(r) =
   \begin{bmatrix}
     1 & 0      & 0       \\
     0 & \cos r & -\sin r \\
     0 & \sin r &  \cos r
   \end{bmatrix}

Step 4 — Combined tag-to-animal rotation matrix
-----------------------------------------------

.. math::

   W = \bigl(R_P(p_0)\; R_R(r_0)\bigr)^\top

When :math:`p_0 = 0` and :math:`r_0 = 0`:

.. math::

   R_P(0) = R_R(0) = I \implies W = I

meaning no rotation is applied and the tag is already aligned with the
animal frame.

Step 5 — Rotate sensor data into the animal frame
-------------------------------------------------

.. math::

   \mathbf{acc}_\text{corr} = \mathbf{acc}\; W

.. math::

   \mathbf{mag}_\text{corr} = \mathbf{mag}\; W

where rows are time samples, so the data matrix is right-multiplied by
:math:`W`.

Step 6 — Pitch and roll in degrees from corrected accelerometer
---------------------------------------------------------------

Let :math:`A_i = \|\mathbf{acc}_{\text{corr},\,i}\|` be the magnitude at
time step :math:`i`. Then:

.. math::

   \text{pitch}_i = -\arcsin\!\left(\frac{a_x^{(i)}}{A_i}\right)
   \cdot \frac{180°}{\pi}

.. math::

   \text{roll}_i = \arctan2\!\left(a_y^{(i)},\; a_z^{(i)}\right)
   \cdot \frac{180°}{\pi}

Step 7 — Gimbal magnetometer to horizontal plane (for heading)
--------------------------------------------------------------

At each time step :math:`i`, undo the pitch and roll to level the
magnetometer vector:

.. math::

   \mathbf{m}_{\text{horiz},\,i} =
   \mathbf{m}_{\text{corr},\,i}\;
   R_R\!\left(\text{roll}_i\right)^\top\;
   R_P\!\left(\text{pitch}_i\right)^\top

Then heading is computed with magnetic declination correction
:math:`\delta`:

.. math::

   \text{heading}_i =
   \arctan2\!\left(m_y^{(i)},\; m_x^{(i)}\right)
   \cdot \frac{180°}{\pi} + \delta
