const Button = ({
  label,
  iconUrl,
  backgroundColor,
  textColor,
  borderColor,
}) => {
  return (
    <button
      className={`flex justify-center items-center gap-2 px-7 py-4 border font-montserrat text-lg leading-none
    ${
      backgroundColor
        ? `${backgroundColor} ${textColor} ${borderColor}`
        : "bg-coral-red text-white border-coral-red"
    } rounded-full ${<img /> ? "hover:${<img />)}" : ""}`}
    >
      {label}
      {iconUrl && (
        <img
          className="ml-2 rounded-full w-5 h-5 hover:translate-x-1"
          src={iconUrl}
          alt="arrow right icon"
        />
      )}
    </button>
  );
};

export default Button;
