import { useState } from "react";
import { IoClose, IoMenu } from "react-icons/io5";
import { NavLink } from "react-router-dom";

const navLinks = [
  {
    label: "Home",
    url: "https://openpolicy.me/",
  },
  {
    label: "Bills",
    url: "/bills",
  },
  {
    label: "Debates",
    url: "/debates",
  },
  {
    label: "MPS",
    url: "/mps",
  },
  {
    label: "Committes",
    url: "/committees",
  },
  {
    label: "About Us",
    url: "https://openpolicy.me/about-us/",
  },
  {
    label: "Admin",
    url: "/admin/scrapers",
  },
  {
    label: "Admin Entities",
    url: "/admin/entities",
  },
];

const Navigation = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <div className="sticky top-0 left-0 right-0 bg-white z-50 w-full shadow-md">
      <div className="padding-x">
        <div className="flex items-center justify-between h-[80px] relative">
          <img
            src="/assets/images/logo.svg"
            alt="logo"
            className="w-[120px] md:w-auto"
          />
          <button
            className="md:hidden z-50"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <IoClose size={24} /> : <IoMenu size={24} />}
          </button>

          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <NavLink
                to={link.url}
                key={link.label + "link"}
                className={({ isActive }) =>
                  isActive
                    ? "font-medium border-b-2 border-black"
                    : "font-medium"
                }
              >
                {link.label}
              </NavLink>
            ))}
          </div>

          <div
            className={`
            fixed top-0 right-0 h-screen w-[70%] bg-white shadow-lg
            transform transition-transform duration-300 ease-in-out
            flex flex-col pt-24 px-6 gap-6 z-40
            md:hidden
            ${isMenuOpen ? "translate-x-0" : "translate-x-full"}
          `}
          >
            {navLinks.map((link) => (
              <NavLink
                to={link.url}
                key={link.label + "mobile-link"}
                className={({ isActive }) =>
                  isActive
                    ? "font-medium border-b-2 border-black"
                    : "font-medium"
                }
                onClick={() => setIsMenuOpen(false)}
              >
                {link.label}
              </NavLink>
            ))}
          </div>

          <button className="hidden md:block bg-black w-[178px] h-[48px] text-white rounded-[8px] cursor-pointer">
            Download Now
          </button>
        </div>
      </div>
    </div>
  );
};

export default Navigation;
