import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CaptionDisplayComponent } from './caption-display.component';

describe('CaptionDisplayComponent', () => {
  let component: CaptionDisplayComponent;
  let fixture: ComponentFixture<CaptionDisplayComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CaptionDisplayComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(CaptionDisplayComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
