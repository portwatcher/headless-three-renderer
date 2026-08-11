pub const SHADER: &str = concat!(
    include_str!("shader/main_bindings.wgsl"),
    include_str!("shader/main_materials.wgsl"),
    include_str!("shader/main_output.wgsl"),
);

pub fn custom_shader_source(fragment_body: &str) -> String {
    CUSTOM_FRAGMENT_SHADER.replace("__CUSTOM_FRAGMENT_BODY__", fragment_body)
}

const CUSTOM_FRAGMENT_SHADER: &str = include_str!("shader/custom_fragment_shader.wgsl");

pub const POST_SHADER: &str = include_str!("shader/post_shader.wgsl");

pub const BACKGROUND_SHADER: &str = include_str!("shader/background_shader.wgsl");
