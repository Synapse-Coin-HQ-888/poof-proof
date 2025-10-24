import numpy as np
from OpenGL import GL as gl

class TexturePool:
    """Handles a pool of reusable OpenGL textures to reduce redundant creation and deletion."""

    def __init__(self):
        self.free_textures: list[int] = []   # Unused texture IDs
        self.active_textures: list[int] = [] # Textures currently in use

    def acquire(self) -> int:
        """Retrieve a texture ID from the pool, creating a new one if none are free."""
        if self.free_textures:
            texture_id = self.free_textures.pop()
        else:
            texture_id = self._generate()
        self.active_textures.append(texture_id)
        return texture_id

    def release(self, texture_id: int):
        """Return a texture ID back to the free pool."""
        if texture_id in self.active_textures:
            self.active_textures.remove(texture_id)
            self.free_textures.append(texture_id)

    def _generate(self) -> int:
        """Generate a new OpenGL texture with default parameters."""
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        return texture_id

    def update_texture(self, texture_id: int, image: np.ndarray):
        """Upload or refresh texture data for the specified texture ID."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

        # Ensure proper RGB format and contiguous array
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Convert float [0,1] arrays to uint8 [0,255]
        if image.dtype in (np.float32, np.float64):
            image = (image * 255).astype(np.uint8)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB8,
            image.shape[1], image.shape[0], 0,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
            image.tobytes()
        )
