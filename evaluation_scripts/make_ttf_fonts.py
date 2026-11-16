"""Convert the OpenType (CFF) files of Linux Libertine into TrueType files that matplotlib embeds as subsetted TrueType
fonts. The family is renamed "Linux Libertine" (without the "O"), as in the TrueType release, so both can be installed
side by side. The files are for your own figures; do not redistribute them.

  python make_ttf_fonts.py <folder with LinLibertine_*.otf> <output folder, e.g. fonts>
"""
import argparse

from pathlib import Path

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable


def convert(source: Path, target: Path) -> None:
    """Converts one CFF font into a TrueType font with quadratic outlines and the renamed family."""
    font = TTFont(source)
    order = font.getGlyphOrder()
    glyphs = font.getGlyphSet()
    glyf = newTable('glyf')
    glyf.glyphOrder = order
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyphs)
        glyphs[name].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
        glyf[name] = pen.glyph()
    font['loca'] = newTable('loca')
    font['glyf'] = glyf
    del font['CFF ']
    if 'VORG' in font:
        del font['VORG']
    maxp = font['maxp']
    maxp.tableVersion = 0x00010000
    for field in ('maxZones', 'maxTwilightPoints', 'maxStorage', 'maxFunctionDefs', 'maxInstructionDefs',
                  'maxStackElements', 'maxSizeOfInstructions', 'maxComponentElements', 'maxComponentDepth',
                  'maxPoints', 'maxContours', 'maxCompositePoints', 'maxCompositeContours'):
        setattr(maxp, field, 1 if field == 'maxZones' else 0)
    post = font['post']
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    for record in font['name'].names:
        if record.nameID in (1, 3, 4, 6, 16):
            text = record.toUnicode()
            record.string = text.replace('LinLibertineO', 'LinLibertine').replace('Libertine O', 'Libertine')
    font.sfntVersion = '\x00\x01\x00\x00'
    target.parent.mkdir(parents=True, exist_ok=True)
    font.save(target)


def main() -> None:
    """Converts the four Linux Libertine styles."""
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='Folder with LinLibertine_*.otf')
    parser.add_argument('target', type=Path, help='Output folder, e.g. fonts')
    args = parser.parse_args()

    # Conversion
    for name in ('LinLibertine_R.otf', 'LinLibertine_RI.otf', 'LinLibertine_RB.otf', 'LinLibertine_RBI.otf'):
        convert(args.source / name, args.target / name.replace('.otf', '.ttf'))
        print('wrote', args.target / name.replace('.otf', '.ttf'))


if __name__ == '__main__':
    main()